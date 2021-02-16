import sys

test_file = sys.argv[1]
open_lines = open(test_file, 'r')
# surnames_lines = open("dist.last.txt", 'r')
forenames_male_lines = open("dist.male.first.txt", 'r')
forenames_female_lines = open("dist.female.first.txt", 'r')
output = open('full-name-output.csv', 'w')

# surnames = []
forenames_female = []
forenames_male = []
first_person_names = []
second_person_names = []
predicted = []
answer = []

# store female forenames
for line in forenames_male_lines:
    line = line.strip()
    arr = line.split(' ')
    forenames_male.append(arr[0])

# store male forenames
for line in forenames_female_lines:
    line = line.strip()
    arr = line.split(' ')
    forenames_female.append(arr[0])

# store surnames
# for line in surnames_lines:
#     line = line.strip()
#     surnames.append(line.split(' ')[0])

# predict names
for line in open_lines:
    line = line.strip()
    [first_person, second_person] = line.split(' AND ')

    # Algorithm to find predicted name
    first_person_words = first_person.split(' ')
    second_person_words = second_person.split(' ')
    if len(first_person_words) == 1:
        if len(second_person_words) == 2:
            predicted_first_person = first_person + " " + second_person_words[-1]
        elif len(second_person_words) == 4:
            predicted_first_person = first_person + " " + second_person_words[-2] + " " + second_person_words[-1]
        else:
            predicted_first_person = first_person + " " + second_person_words[-1]
    elif len(first_person_words) == 2:
        if first_person_words[-1] not in forenames_male and first_person_words[-1] not in forenames_female:
            predicted_first_person = first_person
        elif len(second_person_words) == 4:
            predicted_first_person = first_person + " " + second_person_words[-2] + " " + second_person_words[-1]
        else:
            predicted_first_person = first_person + " " + second_person_words[-1]
    else:
        predicted_first_person = first_person
    # End

    predicted.append(predicted_first_person)
    first_person_names.append(first_person)
    second_person_names.append(second_person)
    output.write(line + ',' + predicted_first_person + '\n')

open_lines.close()
forenames_male_lines.close()
forenames_female_lines.close()
output.close()

answer_lines = open("dev-key.csv", 'r')
for line in answer_lines:
    line = line.strip()
    [feature, ans] = line.split(',')
    answer.append(ans)

# Find Accuracy and Optimization
right = 0
count = 0
count_no_add = 0
count_no_add_right = 0
count_add_1 = 0
count_add_1_right = 0
count_add_2 = 0
count_add_2_right = 0

case_no_error = 0
case_2_error = 0

for i in range(len(answer)):
    predict = predicted[i]
    ans = answer[i]
    if ans == first_person_names[i]:
        if predict == ans:
            count_no_add_right += 1
        count_no_add += 1
    elif ans == first_person_names[i] + " " + second_person_names[i].split(" ")[-1]:
        if predict == ans:
            count_add_1_right += 1

        if len(ans.split(" ")) - len(predict.split(" ")) == 1:
            case_no_error += 1
        if len(ans.split(" ")) - len(predict.split(" ")) == -1:
            case_2_error += 1

        count_add_1 += 1
    else:
        if predict == ans:
            count_add_2_right += 1
        count_add_2 += 1

    if predict == ans:
        right += 1
    else:
        if count <= 10:
            print(i + 1)
            print("First Person: ", first_person_names[i])
            print("Second Person: ", second_person_names[i])
            print("Predicted Value: ", predict)
            print("Answer Value: ", ans)
            count += 1
            continue

print(right, len(answer), "Accuracy:",float(right / len(answer)))
print("No Add: ", count_no_add_right, count_no_add, float(count_no_add_right / count_no_add))
print("Add 1: ", count_add_1_right, count_add_1, float(count_add_1_right / count_add_1))
print("Add 2: ", count_add_2_right, count_add_2, float(count_add_2_right / count_add_2))
print(case_no_error)
print(case_2_error)
