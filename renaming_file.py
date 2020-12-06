import os

# Basic file to rename the data to proper names
def main():

    folder = "../../../Desktop/templeflow_pics_3/train" #For my personal computer
    #folder = "./data/train" # For the github
    
    i = 0
    j = 0
    # turn_left
    cur_fold = os.path.join(folder, "turn_left")
    for count, filename in enumerate(os.listdir(cur_fold)):
        dst = "turn_left" + str(count) + ".jpg"
        src = os.path.join(cur_fold, filename)
        dst = os.path.join(cur_fold, dst)
        os.rename(src, dst);
        i += 1
        j += 1
    
    print(str(i) + " pics in turn_left");
    i = 0
    # turn_right
    cur_fold = os.path.join(folder, "turn_right")
    for count, filename in enumerate(os.listdir(cur_fold)):
        dst = "turn_right" + str(count) + ".jpg"
        src = os.path.join(cur_fold, filename)
        dst = os.path.join(cur_fold, dst)
        os.rename(src, dst);
        i += 1
        j += 1
    
    print(str(i) + " pics in turn_right");
    i = 0
    # lean_left
    cur_fold = os.path.join(folder, "lean_left")
    for count, filename in enumerate(os.listdir(cur_fold)):
        dst = "lean_left" + str(count) + ".jpg"
        src = os.path.join(cur_fold, filename)
        dst = os.path.join(cur_fold, dst)
        os.rename(src, dst);
        i += 1
        j += 1
        
    print(str(i) + " pics in lean_left");
    i = 0
        
    #lean_right
    cur_fold = os.path.join(folder, "lean_right")
    for count, filename in enumerate(os.listdir(cur_fold)):
        dst = "lean_right" + str(count) + ".jpg"
        src = os.path.join(cur_fold, filename)
        dst = os.path.join(cur_fold, dst)
        os.rename(src, dst);
        i += 1
        j += 1
        
    print(str(i) + " pics in lean_right");
    i = 0
    
    # slide
    cur_fold = os.path.join(folder, "slide")
    for count, filename in enumerate(os.listdir(cur_fold)):
        dst = "slide" + str(count) + ".jpg"
        src = os.path.join(cur_fold, filename)
        dst = os.path.join(cur_fold, dst)
        os.rename(src, dst);
        i += 1
        j += 1
        
    print(str(i) + " pics in slide");
    i = 0
        
    # Jump
    cur_fold = os.path.join(folder, "jmp")
    for count, filename in enumerate(os.listdir(cur_fold)):
        dst = "jmp" + str(count) + ".jpg"
        src = os.path.join(cur_fold, filename)
        dst = os.path.join(cur_fold, dst)
        os.rename(src, dst);
        i += 1
        j += 1
        
    print(str(i) + " pics in jmp");
    print(str(j) + " pics total")
        
# Driver Code
if __name__ == '__main__':
    main()
