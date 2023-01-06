# Use Case :
#
# You are on the way to RTO,
#
# they asks your Age
# if your age is greater than or equal to 18 :
# they asks weather you have a driving license or not :
# if you have a driving license , they will asked you to leave.
# if you don't have one , they will first ask about your criminal background ( Y/N question )
# if your age is less than 18 , you are not eligible for taking one
# Create a python program to implement the above use-case wrt the RTO.
# Y/N can't be case sensitive.
# Using a function

def license_checking(age):

    if age > 18 or age == 18:
        lic = input("Do you have your driving license ? (Y/N) ")
        if lic == 'Y' or lic == 'y':
            print("Kindly leave, as re-issuing license is not an option")
        else:
            criminal_background = input("Do you happen to have a criminal background ? (Y/N) ")
            if criminal_background == 'N' or criminal_background == 'n':
                print("you are eligible for driving license")
            else:
                print("You are not eligible for driving license")
    else:
        print("You are not eligible for driving license")


age = int(input("Welcome to RTO, please enter you age: "))
license_checking(age)