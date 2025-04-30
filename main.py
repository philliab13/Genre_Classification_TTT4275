import sys
import subprocess

task_map = {
    1: "Task_1.py",
    2: "Task2.py",
    3: "Task3_embedded_method.py",
    4: "Task3_Wrapper_method.py",
    5: "Task4_80_20.py",
    6: "task4_train_test_split.py"
}


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py [1-6]")
        return

    try:
        task_num = int(sys.argv[1])
    except ValueError:
        print("Input must be a number between 1 and 6.")
        return

    if task_num not in task_map:
        print("Invalid selection. Choose a number from 1 to 6.")
        return

    script_to_run = task_map[task_num]
    try:
        subprocess.run(["python", script_to_run], check=True)
    except subprocess.CalledProcessError:
        print(f"Error: Failed to run {script_to_run}")
    except FileNotFoundError:
        print(f"Error: File {script_to_run} not found.")


if __name__ == "__main__":
    main()
