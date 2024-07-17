import os

total_count = 0
correct_count = 0

for file in os.listdir('output/'):
    with open('output/'+file) as f:
        lines = f.readlines()
        first = lines[0].split(' ')
        prediction = int(first[2])

        if prediction == int(file[-5]): 
            correct_count += 1

    total_count += 1

print("Accuracy : ",100*(correct_count/total_count))