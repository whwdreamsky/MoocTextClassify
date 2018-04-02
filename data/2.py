# coding: utf-8
get_ipython().run_line_magic('run', '1.py')
for child in root[2]:
    if 'answerMatch' in child.attrib:
        fout(root[0].text+'\t'+child.text+'\t'+referans[child.attrib['answerMatch']]+'\t'+child.attrib['accuracy']+'\n')
    else:
        for key,value in referans.iteritem():
            fout(root[0].text+'\t'+child.text+'\t'+value+'\t'+child.attrib['accuracy']+'\n')
            
fout
for child in root[2]:
    if 'answerMatch' in child.attrib:
        fout.write(root[0].text+'\t'+child.text+'\t'+referans[child.attrib['answerMatch']]+'\t'+child.attrib['accuracy']+'\n')
    else:
        for key,value in referans.iteritem():
            fout.write(root[0].text+'\t'+child.text+'\t'+value+'\t'+child.attrib['accuracy']+'\n')
            
for child in root[2]:
    if 'answerMatch' in child.attrib:
        fout.write(root[0].text+'\t'+child.text+'\t'+referans[child.attrib['answerMatch']]+'\t'+child.attrib['accuracy']+'\n')
    else:
        for key,value in referans.iteritems():
            fout.write(root[0].text+'\t'+child.text+'\t'+value+'\t'+child.attrib['accuracy']+'\n')
            

