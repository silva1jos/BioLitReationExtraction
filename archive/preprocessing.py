""" Copy of Preproccessing from script_muazzam """
import os
import sys
import nltk


def getData(foldername):
    """ Gets the data for all files in a folder """
    listOfFiles = os.listdir(foldername)
    # listOfFiles = ['1832467.tsv','2308962.tsv','7593547.tsv'];
    data = []
    for eachFile in listOfFiles:
        if not eachFile.endswith('.tsv'):
            continue
        eachFileContent = []
        fileObj = open(foldername+'/'+eachFile)
        firstLine = 1
        for eachLine in fileObj:
            if firstLine:
                firstLine = 0
                continue
            eachFileContent.append(eachLine.strip())
        fileObj.close()
        data.append(eachFileContent)
        # if len(data) % 35000 == 0:
        # break;
        if len(data) % 1000 == 0:
            print(len(data))
            sys.stdout.flush()

    print(foldername + ' read.')
    return data


def preprocessingTag(data, separator, folderOut):
    dataProcessed = dict()
    noTagCount = 0
    for eachFile in data:
        if eachFile == []:
            noTagCount += 1
            continue
        folderOutObj = open(folderOut+'/'+eachFile[0].split(separator)[0]+'.tsv', 'w')
        folderOutObj.write('PUBMEDID	START	END	VALUE	BIOENTITY	TERM\n')
        for eachLine in eachFile:
            eachLine = eachLine.split(separator)
            if eachLine[0] not in dataProcessed:
                dataProcessed[eachLine[0]] = {'Gene': [], 'Disease': []}
            if (eachLine[4] == 'Gene' or eachLine[4] == 'Disease'):
                eachLine[3] = eachLine[3].replace(' ', '_').upper()
                dataProcessed[eachLine[0]][eachLine[4]].append((eachLine[3], eachLine[1],
                                                                eachLine[2]))
            folderOutObj.write('\t'.join(eachLine) + '\n')
        folderOutObj.close()
    # print(dataProcessed)
    # sys.exit()
    print('No tag count: ' + str(noTagCount))
    print('Preprocessing tag done.')
    return dataProcessed


def preprocessingAbstract(tags, abstracts, separator, folderOut):
    dataProcessed = dict()
    noAbstractCount = 0
    for eachFile in abstracts:
        if eachFile == []:
            continue
        folderOutObj = open(folderOut+'/'+eachFile[0].split(separator)[0]+'.tsv', 'w')
        folderOutObj.write('PUBMEDID	ABSTRACT\n')
        for eachLine in eachFile:
            eachLine = eachLine.split(separator)
            if eachLine[0] not in dataProcessed:
                if len(eachLine) == 2:
                    if eachLine[0] not in tags:
                        continue

                    abst = eachLine[1]
                    for eachTagLine in tags[eachLine[0]]['Disease']:
                        startChar = int(eachTagLine[1])
                        endChar = int(eachTagLine[2])
                        abst = abst.replace(abst[startChar:endChar], eachTagLine[0])
                    for eachTagLine in tags[eachLine[0]]['Gene']:
                        startChar = int(eachTagLine[1])
                        endChar = int(eachTagLine[2])
                        abst = abst.replace(abst[startChar:endChar], eachTagLine[0])
                    folderOutObj.write(eachLine[0] + '\t' + abst + '\n')
                    # dataProcessed[eachLine[0]] = abst;
                else:
                    noAbstractCount += 1
        folderOutObj.close()
    # print(tags)
    print('Files with no tags: ' + str(noAbstractCount))
    print('Abstracts parsed.')
    return 1
    finalOutputProcessed = dict()
    # print(tags['11827742'])
    for eachFile in dataProcessed:
        sentences = ''
        if eachFile not in tags:
            continue
        # listOfDisease = [x for x in tags[eachFile]['Disease']]
        # if not tags[eachFile]['Disease'] == []];
        # listOfGene = [x for x in tags[eachFile]['Gene']]
        for eachSentence in nltk.tokenize.sent_tokenize(dataProcessed[eachFile]):
            # if (any(word in eachSentence for word in listOfDisease) or
            #         any(word in eachSentence for word in listOfGene)):
            #     sentences += eachSentence + ' '
            sentences += eachSentence + ' '
        sentences = sentences[:-1]
        finalOutputProcessed[eachFile] = sentences
    # print(finalOutputProcessed);
    # sys.exit();
    print('Preprocessing abstract done.')
    return finalOutputProcessed


def main():
    ###########################################################################
    # get tag data
    dataTag = getData('../corpus/abstract_tagged')
    sys.stdout.flush()
    dataTag = preprocessingTag(dataTag, '\t', '../corpus/preprocessed_tagged')
    sys.stdout.flush()
    # get abstract data with preprocessing of tag data.
    dataAbstract = getData('../corpus/abstract_only')
    sys.stdout.flush()
    dataAbstract = preprocessingAbstract(dataTag, dataAbstract, '\t', '../corpus/preprocessed')
    sys.stdout.flush()
    # dataAbstract consists of a dictionary. Key: filename, value: sentences
    # that contain both gene and disease from the tag data.
    ###########################################################################
    return 0


if __name__ == '__main__':
    main()
    print('Program Terminated!')
