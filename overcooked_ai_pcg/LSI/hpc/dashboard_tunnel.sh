# Creates an ssh command for connecting to the dashboard.
#
# Usage:
#   dashboard_tunnel.sh SLURM_OUTPUT_FILE
# Example:
#   dashboard_tunnel.sh ./logs/slurm-123456.out

dashboard_line=$(grep "Dashboard Host:" "$1" | head -n 1)
dashboard_addr=$(echo "${dashboard_line}" | sed 's/Dashboard Host:.*|\(.*\)|/\1/g')

echo "Run the following command to open an ssh tunnel from"
echo "localhost:8787 on your local machine to the Dask dashboard:"
echo
echo "  ssh -N -L 8787:$dashboard_addr $USER@discovery.usc.edu"
echo
echo "Make sure to run the above command on your LOCAL machine."
echo
echo "Then visit this address in your browser:"
echo
echo "  http://localhost:8787"
echo
