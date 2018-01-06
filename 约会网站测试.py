def daTest():
    ho = 0.10
    dad,dat = file('daTest.txt')
    no,ra,mi = au(dad)
    m = no.shape[0]
    num = int(m*ho)
    err = 0.0
    for i in range(num):
	result = classify0(no[i,:],no[num:m,:],dat[num,m],3)
	print "the classifier came back with: %d,the real answer is: %d" % (result,dat[i])
	if (result != dat[i]): err += 1.0
	print "the total error rate is: %f" % (err / float(num))

