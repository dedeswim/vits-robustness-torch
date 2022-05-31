gcloud alpha compute tpus tpu-vm ssh vits-robustness-pod --worker=all --command="sudo usermod -a -G docker \${USER}"
gcloud alpha compute tpus tpu-vm ssh vits-robustness-pod --worker=all --command="gcloud auth configure-docker gcr.io -y"
gcloud alpha compute tpus tpu-vm ssh vits-robustness-pod --worker=all --command="docker pull gcr.io/vits-robustness/vits-robustness:latest"
gcloud alpha compute tpus tpu-vm ssh vits-robustness-pod --worker=all --command="mkdir -p ~/output/"
while ! [ -f /tmp/libtpu.so ]; do
    sleep 30
done

echo "Set-up complete!"