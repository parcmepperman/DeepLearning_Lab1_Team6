import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

# Bitcoin market report from 2018

loan_data = "BTC.xls"

# reading in of .xls data
loan_book = xlrd.open_workbook(loan_data, encoding_override="uft-8")
loan_sheet = loan_book.sheet_by_index(0)
data = np.asarray([[loan_sheet.row_values(i)[3], loan_sheet.row_values(i)[4]]
                   for i in range(1, loan_sheet.nrows)])
n_samples = loan_sheet.nrows

# input 1 variables (Lowest daily value(x) & adjusted closing value(y))
X = tf.placeholder(tf.float32, name='Low')
Y = tf.placeholder(tf.float32, name='Close')

# creating weights and biases
weight = tf.Variable(0.0, name='weights')
biases = tf.Variable(0.0, name='bias')

# model for predicting y
Y_predict = X * weight + biases
loss = tf.square(Y - Y_predict, name='loss')

# optimizing learning_rate to get the minimum loss, 0.1 * 10^-9 best fit line and least amount of loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000000001).minimize(loss)
init = tf.initialize_all_variables()

with tf.Session() as sess:

    sess.run(init)
    tf.summary.scalar('Prediction', Y_predict)
    tf.summary.scalar('Loss', loss)
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('/Users/marcpepperman/Desktop/DL_Lab_1_Team6_Student24/graphs/linear_reg',
                                           graph=tf.get_default_graph())

    # increase range to insure best possible regression line
    for i in range(3000):
        total_loss = 0
        for xs, ys in data:
            _, l, summary = sess.run([optimizer, loss, merged_summary], feed_dict={X: xs, Y: ys})
            summary_writer.add_summary(summary, i)
            total_loss += l

        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))

    weight, biases = sess.run([weight, biases])

X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real Data')
plt.plot(X, X * weight + biases, 'r', label='Predicted Data')

plt.legend()
plt.show()
plt.savefig('DL_Lab_Number_One.png')
