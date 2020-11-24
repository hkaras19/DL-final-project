import os

# Basic file to rename the data to proper names
def main():

    folder = "../../../Desktop/TEMPLEFLOW_PICS" #For my personal computer
    data_folder = "../data" # For the github

    # turn_left
    cur_fold = os.path.join(folder, "turn_left")
    for count, filename in enumerate(os.listdir(cur_fold)):
        dst = "turn_left" + str(count) + ".png"
        src = os.path.join(cur_fold, filename)
        dst = os.path.join(cur_fold, dst)
        os.rename(src, dst);

    # turn_right
    cur_fold = os.path.join(folder, "turn_right")
    for count, filename in enumerate(os.listdir(cur_fold)):
        dst = "turn_right" + str(count) + ".png"
        src = os.path.join(cur_fold, filename)
        dst = os.path.join(cur_fold, dst)
        os.rename(src, dst);
        
    # lean_left
    cur_fold = os.path.join(folder, "lean_left")
    for count, filename in enumerate(os.listdir(cur_fold)):
        dst = "lean_left" + str(count) + ".png"
        src = os.path.join(cur_fold, filename)
        dst = os.path.join(cur_fold, dst)
        os.rename(src, dst);
        
    #lean_right
    cur_fold = os.path.join(folder, "lean_right")
    for count, filename in enumerate(os.listdir(cur_fold)):
        dst = "lean_right" + str(count) + ".png"
        src = os.path.join(cur_fold, filename)
        dst = os.path.join(cur_fold, dst)
        os.rename(src, dst);
        
    # slide
    cur_fold = os.path.join(folder, "slide")
    for count, filename in enumerate(os.listdir(cur_fold)):
        dst = "slide" + str(count) + ".png"
        src = os.path.join(cur_fold, filename)
        dst = os.path.join(cur_fold, dst)
        os.rename(src, dst);
        
    # Jump
    cur_fold = os.path.join(folder, "jmp")
    for count, filename in enumerate(os.listdir(cur_fold)):
        dst = "jmp" + str(count) + ".png"
        src = os.path.join(cur_fold, filename)
        dst = os.path.join(cur_fold, dst)
        os.rename(src, dst);
        
# Driver Code
if __name__ == '__main__':
    main()
