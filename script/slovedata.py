# coding: utf-8
import xml.etree.ElementTree as ET
import sys


def transdatafromXML (f_indexqueryfile,f_relationfile,inputxml):
    tree = ET.parse(inputxml)
    root = tree.getroot()

    referans = {}
    for child in root[1]:
        referans[child.attrib['id']] = child.text
        f_indexqueryfile.write(child.attrib['id']+'\t'+child.text+'\n')

    for child in root[2]:
        if 'answerMatch' in child.attrib:
            f_relationfile.write(root.attrib['id'] + '\t' + child.attrib['id'] + '\t' + child.attrib['answerMatch'] + '\t' \
                       + child.attrib['accuracy'] + '\n')
        else:
            for key, value in referans.items():
                f_relationfile.write(root.attrib['id'] + '\t' + child.attrib['id'] + '\t' + key + '\t' + child.attrib['accuracy'] + '\n')
        f_indexqueryfile.write(child.attrib['id']+'\t'+child.text+'\n')

    f_indexqueryfile.write(root.attrib['id']+'\t'+root[0].text+'\n')

    



if __name__ == '__main__':

    indexqueryfile,relationfile = sys.argv[1:]
    f_indexqueryfile = open(indexqueryfile,'a')
    f_relationfile = open(relationfile,'a')
    while True:
        inputxml = sys.stdin.readline()
        if not inputxml:
            f_indexqueryfile.close()
            f_relationfile.close()
            break;
        transdatafromXML(f_indexqueryfile,f_relationfile,inputxml.strip())

