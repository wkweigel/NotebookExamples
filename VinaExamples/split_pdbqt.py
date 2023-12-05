import os

def split_pdbqt(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    pose_separator = "MODEL"
    current_structure = []
    structure_count = 0

    for line in lines:
        if line.startswith(pose_separator):
            if current_structure:
                # Save the current structure to a separate file
                output_filename = f"structure_{structure_count}.pdbqt"
                with open(output_filename, 'w') as output_file:
                    output_file.writelines(current_structure)
                current_structure = []
                structure_count += 1
        current_structure.append(line)

    # Save the last structure
    if current_structure:
        output_filename = f"structure_{structure_count}.pdbqt"
        with open(output_filename, 'w') as output_file:
            output_file.writelines(current_structure)

if __name__ == "__main__":
    input_pdbqt_file = "your_input.pdbqt"
    split_pdbqt(input_pdbqt_file)
    print("Splitting complete.")