def plotline(params, steps=100, axislim=[0, 100, 0, 100], title="None", labels="None", legend="None"):
    xmin, xmax, ymin, ymax = axislim
    x = np.linspace(xmin, xmax, steps)
    xmatrix = []
    for n in range(len(params)):
        xmatrix.append(x ** n)
    xmatrix = np.matrix(xmatrix)
    params = params[::-1]
    y = np.matmul(params, xmatrix)
    if title!="None":
        plt.title(title)
    if legend!="None":
        plt.legend(legend)
    axes = plt.gca()
    axes.set_xlim([xmin,xmax])
    axes.set_ylim([ymin,ymax])
    if labels!="None":
        axes.set_xlabel(labels[0])
        axes.set_ylabel(labels[1])
    plt.plot(x, np.transpose(y))
    plt.show()
        
def lsq(params, points):
    params = np.array(params)
    points = np.array(points)
    params = params[::-1]
    x = points[:, 0]
    xmatrix = []
    for n in range(len(params)):
        xmatrix.append(x ** n)
    xmatrix = np.matrix(xmatrix)
    cost = np.sum(abs(np.matmul(params, xmatrix) - points[:, 1]))
    return cost

points=[[0,0], [1,1], [-1,1], [-2, -8]]
fit = minimize(lsq, np.array([1, -1, -0.04, 0.76]), method='Nelder-Mead', args=(points))
plotline(fit.x, steps=100, axislim=[-4,4,-10,10], 
         title="Model Difference in capital letters", 
         labels=["Difference in capital letters", "Percent chance"])