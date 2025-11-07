import os
import random
import numpy as np

data = []

random.seed()

def extractInfoFromFileName(fileName):
    id = ''
    gender = ''
    age = ''
    topic = ''
    zodiacSign = ''

    readID = False
    readGender = False
    readAge = False
    readTopic = False
    readZodiacSign = False

    for char in fileName:
        if readID:
            if char == '.':
                readID = False
                readGender = True
                continue

            else:
                id += char
                continue

        elif readGender:
            if char == '.':
                readGender = False
                readAge = True
                continue

            else:
                gender += char
                continue

        elif readAge:
            if char == '.':
                readAge = False
                readTopic = True
                continue

            else:
                age += char
                continue

        elif readTopic:
            if char == '.':
                readTopic = False
                readZodiacSign = True
                continue

            else:
                topic += char
                continue

        elif readZodiacSign:
            if char == '.':
                break

            else:
                zodiacSign += char
                continue

        elif char == "/":
            readID = True

    return id, gender, int(age), topic, zodiacSign
            
i = 0

blogInfo = []

for file in os.listdir('blogs'):
    fileName = 'blogs/' + file
    if os.path.isfile(fileName):

        postCount = 0
        postTag = '<post>'
        curTag = ''

        id, gender, age, topic, zodiacSign = extractInfoFromFileName(fileName)

        data.append({})

        data[i]['fileName'] = fileName
        data[i]['id'] = id
        data[i]['gender'] = gender
        data[i]['age'] = age
        data[i]['topic'] = topic
        data[i]['zodiacSign'] = zodiacSign

        fileContents = open(fileName, 'r', errors='ignore') #, encoding='utf-8'
        # NOTE THIS TEXT TIME I SPEAK TO MILTON
        # GIVING UnicodeDecodeError when errors aren't ignored
        # The encoding='utf-8' parameter doesn't seem to work

        while True:
            char = fileContents.read(1)

            if not char:
                break

            if postTag != '':
                curTag += char
                if len(curTag) == len(postTag):
                    if curTag == postTag:
                        postCount += 1
                    curTag = ''
                    continue
                else:
                    if curTag != postTag[:len(curTag)]:
                        curTag = ''
                        continue
            else:
                if char == '<':
                    curTag = '<'


        blogInfo.append(postCount)
        data[i]['postCount'] = postCount

    i += 1

    if i % 500 == 0:
        print(i)

blogInfo = np.array(blogInfo)

usersWithMostPosts = []

# ADJUSTABLE PARAMETER TO DECIDE HOW MANY OF TOP
# FREQUENT POSTERS WE WANT
for i in range(1000):
    usersWithMostPosts.append(['', 0])

for user in data:
    if user['postCount'] > usersWithMostPosts[999][1]:
        for j in range(len(usersWithMostPosts)):
            if user['postCount'] > usersWithMostPosts[j][1]:
                usersWithMostPosts.insert(j, [user['fileName'], user['postCount']])
                usersWithMostPosts.pop()
                break

with open('usersWithMostPosts.txt', 'w') as outfile:
    outfile.write('Users with Most Posts\n\n')
    outfile.write('File name\tPostCount\n')
    for user in usersWithMostPosts:
        outString = user[0] + '\t' + str(user[1]) + '\n'
        outfile.write(outString)


# print(len(blogInfo))
# print(np.average(blogInfo))
# print(np.min(blogInfo))
# print(np.max(blogInfo))
# print(np.median(blogInfo))
# print(np.sum(blogInfo))

# genderCounts = {}
# topicCounts = {}

# for user in data:
#     if user['gender'] in genderCounts:
#         genderCounts[user['gender']] += 1
#     else:
#         genderCounts[user['gender']] = 1
    
#     if user['topic'] in topicCounts:
#         topicCounts[user['topic']] += 1
#     else:
#         topicCounts[user['topic']] = 1

# with open('genderCounts.txt', 'w') as outfile:
#     for key in genderCounts:
#         outString = key + '\t' + str(genderCounts[key]) + '\n'
#         outfile.write(outString)

# topics = []

# for i in range(100):
#     topics.append(['', 0])

# for topic in topicCounts:
#     if topicCounts[topic] > topics[99][1]:
#         for j in range(len(topics)):
#             if topicCounts[topic] > topics[j][1]:
#                 topics.insert(j, [topic, topicCounts[topic]])
#                 topics.pop()
#                 break

# with open('topicCounts.txt', 'w') as outfile:
#     outfile.write('Most Frequent Topics\n\n')
#     outfile.write('Topic\tCount\n')
#     for tup in topics:
#         outString = tup[0] + '\t' + str(tup[1]) + '\n'
#         outfile.write(outString)







