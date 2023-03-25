import argparse
import os
import pathlib
current_file_location = str(pathlib.Path(__file__).parent.resolve())

PROJECT_FOLDER = current_file_location.split("/")[-1]
parser = argparse.ArgumentParser()
parser.add_argument("-py", "--py_func", required=True, type=str)
parser.add_argument("-l", "--model-lang", required=True, type=str, nargs="+")
parser.add_argument("-v", "--variable", type=str)
args = parser.parse_args()
variable_name = None
variable_value = None
if args.variable:
    variable_name = args.variable.split("=")[0]
    variable_value = args.variable.split("=")[1]

job_prefix = args.py_func.split(".")[0]
# write sh file
for mlang in args.model_lang:
    with open(f"./{mlang}-{job_prefix}.sh", "w") as f:
        if variable_value:
            f.write(
                f"""#!/bin/bash
    echo "Activating huggingface environment"
    source /share/apps/anaconda3/2021.05/bin/activate huggingface
    echo "Beginning script"
    cd /share/luxlab/andrea/{PROJECT_FOLDER}
    python3 {args.py_func} --model-lang {mlang}
                """
            )
        else:
            f.write(
                f"""#!/bin/bash
                echo "Activating huggingface environment"
                source /share/apps/anaconda3/2021.05/bin/activate huggingface
                echo "Beginning script"
                cd /share/luxlab/andrea/{PROJECT_FOLDER}
                python3 {args.py_func} --model-lang {mlang} --{variable_name} {variable_value}
                            """
        )

    with open(f"./{mlang}-{job_prefix}.sub", "w") as f:
        f.write(
            f"""#!/bin/bash
#SBATCH -J {mlang}-{job_prefix}                          # Job name
#SBATCH -o /share/luxlab/andrea/{PROJECT_FOLDER}/logs/{mlang}-{job_prefix}_%j.out # output file (%j expands to jobID)
#SBATCH -e /share/luxlab/andrea/{PROJECT_FOLDER}/logs/{mlang}-{job_prefix}_%j.err # error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                        # Request status by email
#SBATCH --mail-user=aww66@cornell.edu          # Email address to send results to.
#SBATCH -N 1                                   # Total number of nodes requested
#SBATCH -n 8                                  # Total number of cores requested
#SBATCH --get-user-env                         # retrieve the users login environment
#SBATCH --mem=50G                             # server memory requested (per node)
#SBATCH -t 10:00:00                            # Time limit (hh:mm:ss)
#SBATCH --partition=gpu          # Request partition
#SBATCH --gres=gpu:1
/share/luxlab/andrea/{PROJECT_FOLDER}/{mlang}-{job_prefix}.sh
            """
        )
    os.system(f"chmod 775 {mlang}-{job_prefix}.sh")
    os.system(f"sbatch --requeue {mlang}-{job_prefix}.sub")


