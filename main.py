# Student Grading System

#Function tp calculate grade of student
def calculate_grade(average):
    if average >= 90:
        return "A"
    elif average >= 75:
        return "B"
    elif average >= 50:
        return "C"
    else:
        return "F"


#Main  function
def main():
    print("------Student Grading System-----")
    student_name = input("Enter Student Name: ")
    student_class = input("Enter Class: ")

    #Taking number of subjects with validation
    while True:
        try:
            num_subjects = int(input("Enter Number of Subjects: "))

            if num_subjects >0:
                break
            else:
                print("Number of subjects must be greater than 0.")

        except ValueError:
            print("Invalid input! Please enter a whole number.")

    subjects = {}
    total_marks = 0


    #Taking each subject marks
    for i in range(num_subjects):
        subject = input(f"\nEnter Subject {i + 1} Name: ")

        while True:
            try:
                marks = float(input(f"Enter Marks for {subject} (0-100): "))

                if 0 <= marks <= 100:
                    break
                else:
                    print("Marks must be between 0 and 100.")

            except ValueError:
                print("Invalid input! Please enter a valid number.")

        subjects[subject] = marks
        total_marks += marks

    average = total_marks / num_subjects
    grade = calculate_grade(average)

    #Printing Result
    print("\n-------RESULT-------")
    print(f"Student Name : {student_name}")
    print(f"Class        : {student_class}")

    print("\nMARKS :")
    for subject, marks in subjects.items():
        print(f"{subject}: {marks}")

    print(f"\nTotal Marks : {total_marks}")
    print(f"Average     : {average:.2f}")
    print(f"Grade       : {grade}")



if __name__ == "__main__":
    main()