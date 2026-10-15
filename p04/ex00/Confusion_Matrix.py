import sys
import matplotlib.pyplot as plt

def get_matrix(prediction_fd, truth_fd):
	mat = [[0,0],[0,0]]
	values = {"Jedi": 0, "Sith": 1}
	val1 = 0
	val2 = 0

	for line1, line2 in zip(prediction_fd, truth_fd):
		line1 = line1.strip()
		line2 = line2.strip()
		try:
			val1 = values[line1]
		except KeyError:
			print(f"`{line1}' is not a valid option, only Jedi and Sith are")
			return None
		try:
			val2 = values[line2]
		except KeyError:
			print(f"`{line2}' is not a valid option, only Jedi and Sith are")
			return None

		mat[val2][val1] += 1

	return mat

def describe(mat):
	tp = mat[0][0]
	fn = mat[0][1]
	fp = mat[1][0]
	tn = mat[1][1]

	total = tn + tp + fn + fp

	tprecision = 0
	try:
		tprecision = tp / (tp + fp)
	except:
		tprecision = 1

	trecall = 0
	try:
		trecall = tp / (tp + fn)
	except:
		trecall = 1

	nprecision = 0
	try:
		nprecision = tn / (tn + fn)
	except:
		nprecision = 1

	nrecall = 0

	try:
		nrecall = tn / (tn + fp)
	except:
		nrecall = 1

	tf1 = 2 * tprecision * trecall / (tprecision + trecall)
	nf1 = 2 * nprecision * nrecall / (nprecision + nrecall)

	acc = (tp + tn) / total

	print(f"{'':10}{'precision':10}{'recall':10}{'f1-score':10}{'total':10}\n")
	print(f"{'Jedi':10}{tprecision:<10.2f}{trecall:<10.2f}{tf1:<10.2f}{sum(mat[0]):<10}")
	print(f"{'Sith':10}{nprecision:<10.2f}{nrecall:<10.2f}{nf1:<10.2f}{sum(mat[1]):<10}\n")
	print(f"{'accuracy':10}{'':10}{'':10}{acc:<10.2f}{total:<10}\n")
	print(f"[[{tp} {fp}]\n [{fn} {tn}]]")

def plot(mat):
	plt.imshow(mat, cmap="Blues")

	for i in range(2):
		for j in range(2):
			plt.text(j, i, mat[i][j],
						ha="center",
						va="center")

	plt.colorbar()

	plt.xticks([0, 1], ["Jedi", "Sith"])
	plt.yticks([0, 1], ["Jedi", "Sith"])

	plt.xlabel("Predicted")
	plt.ylabel("Actual")

	plt.show()

def main():
	if len(sys.argv) != 3:
		print("Bad imput, usage: python3 Confusion_Matrix predictions_file truth_file")
		return

	with open(sys.argv[1]) as fd:
		l = 0
		for _ in fd:
			l += 1
		with open(sys.argv[2]) as fd2:
			l2 = 0
			for _ in fd2:
				l2 += 1
			if l != l2:
				print("Bad imput, files given don't have the same amount of lines")
				return
	with open(sys.argv[1]) as fd:
		with open(sys.argv[2]) as fd2:
			mat = get_matrix(fd, fd2)
			if not mat:
				return
			describe(mat)
			plot(mat)

if __name__ == "__main__":
	main()
