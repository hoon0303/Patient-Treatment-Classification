from matplotlib import pyplot as plt

topics = ['Logistic Model', 'DecisionTree', 'RandomForest']
value_a = [0.7223271628258405, 0.7492447129909365, 0.9996222138269739]
value_b = [0.7069486404833837, 0.7175226586102719, 0.7507552870090635]
def create_x(t, w, n, d):
    return [t*x + w*n for x in range(d)]
value_a_x = create_x(2, 0.8, 1, 3)
value_b_x = create_x(2, 0.8, 2, 3)
ax = plt.subplot()
ax.bar(value_a_x, value_a)
ax.bar(value_b_x, value_b)
middle_x = [(a+b)/2 for (a,b) in zip(value_a_x, value_b_x)]
ax.set_xticks(middle_x)
ax.set_xticklabels(topics)
plt.title('Machine Learning Score')
plt.ylim([0.5, 1]) 
plt.show()