import ImageProcessing as IP




#IP.CompressImage_FixSize(FileName)
#print 'a'
#IP.CompressImage_Template(FileName, 1)
#IP.CompressImage_Template(FileName, 2)
#print 'b'
#IP.VerticalFlip(FileName)
#print 'c'
#IP.MergetoTemplate(FileName, 1, Position)
#IP.MergetoTemplate(FileName, 2, Position)
#print 'd'
#IP.EyelidAndEyeCornerSegmetation(FileName, Position)
#print 'e'
#IP.IrisSegmetation(FileName, Position)
#print 'f'

#IP.ScleraSegmentation(FileName, Position)
#print 'g'
# 4, 9, 19, 44, 54, 60, 89, 91, 96, 97, 100, 103, 108, 115
#for a in range(10, 11):
#    print a
#    IP.DrawOutline('Case '+str(a)+'.jpg', 2)

#for a in range(116, 127):
#    print a
#    IP.DrawOutline('Case '+str(a)+'.jpg', 1)

FileName = 'Case 37.jpg'
Position = 2



for a in range (1, 57):
    if a == 4 or a == 9 or a == 18 or a == 44 or a == 54:
        continue

    FileName = 'Case '+str(a)+'.jpg'
    print FileName
    IP.ScleraSegmentation(FileName, Position)
    IP.RegiontoNumber()

Position = 1

for a in range (57, 127):
    if a == 60 or a == 89 or a == 91 or a == 96 or a == 97 or a == 100 or a == 103 or a == 108 or a == 115:
        continue

    FileName = 'Case '+str(a)+'.jpg'
    print FileName
    IP.ScleraSegmentation(FileName, Position)
    IP.RegiontoNumber()
