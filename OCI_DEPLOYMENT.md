# Free deployment on Oracle Cloud Infrastructure (OCI)

This guide deploys the Flask app to an OCI Always Free Ampere A1 Ubuntu VM. The app runs inside Docker on the VM. The VM is Linux; your own computer can remain Windows and connect through PowerShell's OpenSSH client.

OCI's current Always Free A1 allowance for a free-only tenancy is up to 2 OCPUs and 12 GB RAM total. This is much more memory than Render Free's 512 MB and should fit the TensorFlow model. TensorFlow 2.21 has a Python 3.11 Linux ARM64 wheel, which matches the A1 processor. OCI can reclaim VMs that remain very idle for seven days, and A1 capacity may not be available in every region at all times.

Before starting the VM setup, save and push the Oracle deployment files from PowerShell in the project folder:

```powershell
git add Dockerfile .dockerignore OCI_DEPLOYMENT.md README.md
git commit -m "Prepare Oracle free deployment"
git push
```

If Git says there is nothing to commit, continue. The later `git archive` command packages the latest commit, including the model tracked in the repository.

## 1. Create an OCI account and VM

1. Create or sign in to an Oracle Cloud account and choose the **Always Free** account option. Follow Oracle's account verification steps. Do not upgrade to Pay As You Go for this guide.
2. Open the OCI Console and select your **Home region**.
3. Go to **Compute → Instances → Create instance**.
4. Name it `banana-classifier`.
5. For the image, choose an Ubuntu 22.04 or newer **ARM64/AArch64** image.
6. For the shape, choose **Ampere A1 Flex** and set **2 OCPUs** and **12 GB memory**. Confirm that OCI marks the instance as **Always Free eligible** before creating it.
7. Use a public subnet and assign a public IPv4 address.
8. Generate an SSH key pair in OCI and download the private key to your Windows computer. Keep the private key private; do not upload it to GitHub or share it.
9. Create the instance, then copy its public IPv4 address from the instance details page.

If OCI reports that the host has insufficient capacity, try another availability domain in the home region or retry later. Do not select a paid shape to get around the capacity message.

## 2. Allow web traffic to the VM

In the VM's subnet security list (or network security group), add inbound rules:

- TCP port `22` from your current public IP address only (SSH administration).
- TCP port `8080` from `0.0.0.0/0` (public demo website).

Keep SSH restricted to your own IP. Port 8080 serves plain HTTP and is intended for a short student demonstration, not a production service that handles sensitive uploads.

## 3. Connect from Windows PowerShell

Open PowerShell on Windows and replace the two example paths/values with your actual private-key path and VM address:

```powershell
ssh -i "$env:USERPROFILE\Downloads\oci-private-key.key" ubuntu@YOUR_VM_PUBLIC_IP
```

If Windows says the private key permissions are too open, restrict the key file's permissions in Windows Properties → **Security**, or use:

```powershell
icacls "$env:USERPROFILE\Downloads\oci-private-key.key" /inheritance:r
icacls "$env:USERPROFILE\Downloads\oci-private-key.key" /grant:r "$($env:USERNAME):(R)"
```

## 4. Install Docker on the Ubuntu VM

Run these commands in the SSH session:

```bash
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu
exit
```

Reconnect from PowerShell using the same `ssh` command so the new Docker group membership takes effect.

## 5. Copy the committed app from Windows to the VM

In a second PowerShell window, go to the project folder (the folder containing `Dockerfile`) and create a source archive from the committed files:

```powershell
git archive --format=tar --output=banana-app.tar HEAD
scp -i "$env:USERPROFILE\Downloads\oci-private-key.key" .\banana-app.tar ubuntu@YOUR_VM_PUBLIC_IP:/home/ubuntu/
```

Back in the SSH session, unpack it and build the container:

```bash
mkdir -p ~/banana-app
tar -xf ~/banana-app.tar -C ~/banana-app
cd ~/banana-app
docker build -t banana-classifier .
```

The Docker image contains the trained model and inference files. The `.dockerignore` file keeps datasets, reports, tests, and training artifacts out of the container build context.

## 6. Start the website

Run the container and check that it stays up:

```bash
docker run -d --name banana-classifier --restart unless-stopped -p 8080:8080 banana-classifier
docker ps
docker logs --tail 100 banana-classifier
```

The first start may take a few minutes while TensorFlow loads and warms up the model. Then open this address in a browser, replacing the placeholder with the VM's public address:

```text
http://YOUR_VM_PUBLIC_IP:8080
```

The health endpoint should return `{"status":"ok"}` at `http://YOUR_VM_PUBLIC_IP:8080/health`.

## 7. Update the running app later

Commit and push changes from Windows, then create and transfer a fresh archive using step 5. On the VM, rebuild and replace the container:

```bash
cd ~/banana-app
docker build -t banana-classifier .
docker rm -f banana-classifier
docker run -d --name banana-classifier --restart unless-stopped -p 8080:8080 banana-classifier
```

## Free-tier and security notes

- Keep the shape within the Always Free allowance. Oracle's current documentation says an Always Free tenancy's Ampere A1 allowance is 2 OCPUs and 12 GB RAM total.
- Oracle may reclaim an Always Free compute instance that meets its idle criteria for seven days. See [Oracle Always Free resources](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm).
- The VM has a public address and port 8080 is plain HTTP. Don't use it for private or sensitive images. For a longer-term public service, add HTTPS and review access controls.
- History is disabled in the container; uploaded images are processed in memory and are not written to disk by the app.
