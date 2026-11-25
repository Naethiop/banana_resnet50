"""Near-duplicate photograph detection shared by the audit (Stage 3) and data preparation (Stage 4).

Two images are treated as the same photograph when their 48x48 greyscale versions are strongly
correlated after removing brightness and contrast, under any of the 8 flips and 90-degree rotations.
This is a heuristic: heavily cropped or rescaled copies can be missed.
"""
import numpy as np
from PIL import Image, ImageFilter

THUMB_SIZE = 48


def thumbnail(img):
    """Greyscale 48x48, lightly blurred, as a float array."""
    g = img.convert("L").resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS).filter(ImageFilter.GaussianBlur(1))
    return np.asarray(g, dtype=float)


def normalise(a):
    v = np.asarray(a, dtype=float).flatten()
    v = (v - v.mean()) / (v.std() + 1e-6)
    return v / np.sqrt(v.size)


def dihedral(a):
    """The 8 flips/rotations of a 2-D array."""
    for k in range(4):
        r = np.rot90(a, k)
        yield r
        yield np.fliplr(r)


def similarity_from_thumbnails(thumbs):
    """Symmetric n x n similarity matrix (diagonal set to -1)."""
    base = np.array([normalise(a) for a in thumbs])
    sim = np.full((len(thumbs), len(thumbs)), -1.0)
    for t in range(8):
        vt = np.array([normalise(list(dihedral(a))[t]) for a in thumbs])
        sim = np.maximum(sim, base @ vt.T)
    sim = np.maximum(sim, sim.T)
    np.fill_diagonal(sim, -1.0)
    return sim


def similarity_from_paths(paths):
    thumbs = []
    for p in paths:
        with Image.open(p) as im:
            thumbs.append(thumbnail(im))
    return similarity_from_thumbnails(thumbs)


def union_find_groups(sim, threshold):
    """Group ids (0..k-1) after linking every pair with similarity >= threshold."""
    n = sim.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ii, jj = np.where(np.triu(sim >= threshold, k=1))
    for i, j in zip(ii, jj):
        parent[find(int(i))] = find(int(j))
    roots = [find(i) for i in range(n)]
    ids = {r: k for k, r in enumerate(sorted(set(roots)))}
    return [ids[r] for r in roots]
