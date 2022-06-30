from sklearn.metrics import confusion_matrix, classification_report

def trainingAndTest(classifier, train_x,train_y,test_x,test_y):
    classifier.fit(train_x,train_y)
    results_y = classifier.predict(test_x)
    print(classification_report(test_y, results_y))
    cf_matrix = confusion_matrix(test_y, results_y)
    print(cf_matrix)
    # We plot the matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])
    # Display the visualization of the Confusion Matrix.
    plt.show()