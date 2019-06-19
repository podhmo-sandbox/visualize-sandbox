from nbreversible import code
with code():
    import pandas as pd # ライブラリ pandas をインポートし、以下 pd と呼ぶことにする。


with code():
    # https://www.kaggle.com/abcsds/pokemon から取得した Pokemon.csv を読み込む。
    df = pd.read_csv("Pokemon.csv") # df とは、 pandas の DataFrame 形式のデータを入れる変数として命名


with code():
    df.head() # 先頭５行を表示


with code():
    df.head(50).style.bar(subset=['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'])


with code():
    df.describe() # 平均値、標準偏差、最小値、25%四分値、中央値、75%四分値、最大値をまとめて表示


with code():
    # Jupyter 上で絵を表示するためのマジックコマンド
    #%matplotlib inline 

    import matplotlib.pyplot as plt # matplotlib.pyplot をインポートし、以下 plt と呼ぶ。


with code():
    # 散布図行列
    from pandas.tools import plotting
    # from pandas import plotting # 新しいバージョンではこちらを
    plotting.scatter_matrix(df.iloc[:, 4:11], figsize=(8, 8))
    plt.show()


with code():
    type1s = list(set(list(df['Type 1'])))


with code():
    print(len(type1s), type1s)


with code():
    cmap = plt.get_cmap('coolwarm')
    colors = [cmap((type1s.index(c) + 1) / (len(type1s) + 2)) for c in df['Type 1'].tolist()]
    plotting.scatter_matrix(df.iloc[:, 4:11], figsize=(8, 8), color=colors, alpha=0.5)
    plt.show()


with code():
    from matplotlib.colors import LinearSegmentedColormap
    dic = {'red':   ((0, 0, 0), (0.5, 1, 1), (1, 1, 1)),
           'green': ((0, 0, 0), (0.5, 1, 1), (1, 0, 0)),
           'blue':  ((0, 1, 1), (0.5, 0, 0), (1, 0, 0))}

    tricolor_cmap = LinearSegmentedColormap('tricolor', dic)


with code():
    cmap = tricolor_cmap
    colors = [cmap((type1s.index(c) + 1) / (len(type1s) + 2)) for c in df['Type 1'].tolist()]
    plotting.scatter_matrix(df.iloc[:, 4:11], figsize=(8, 8), color=colors)
    plt.show()


with code():
    import numpy as np
    pd.DataFrame(np.corrcoef(df.iloc[:, 4:11].T.values.tolist()),
                 columns=df.iloc[:, 4:11].columns, index=df.iloc[:, 4:11].columns)


with code():
    corrcoef = np.corrcoef(df.iloc[:, 4:11].T.values.tolist())
    plt.imshow(corrcoef, interpolation='nearest', cmap=plt.cm.coolwarm)
    plt.colorbar(label='correlation coefficient')
    tick_marks = np.arange(len(corrcoef))
    plt.xticks(tick_marks, df.iloc[:, 4:11].columns, rotation=90)
    plt.yticks(tick_marks, df.iloc[:, 4:11].columns)
    plt.tight_layout()


with code():
    corrcoef = np.corrcoef(df.iloc[:, 4:11].T.values.tolist())
    plt.imshow(corrcoef, interpolation='nearest', cmap=tricolor_cmap)
    plt.colorbar(label='correlation coefficient')
    tick_marks = np.arange(len(corrcoef))
    plt.xticks(tick_marks, df.iloc[:, 4:11].columns, rotation=90)
    plt.yticks(tick_marks, df.iloc[:, 4:11].columns)
    plt.tight_layout()


with code():
    # バイオリンプロット
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.violinplot(df.iloc[:, 4:11].values.T.tolist())
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7]) #データ範囲のどこに目盛りが入るかを指定する
    ax.set_xticklabels(df.columns[4:11], rotation=90)
    plt.grid()
    plt.show()


with code():
    for index, type1 in enumerate(type1s):
        df2 = df[df['Type 1'] == type1]
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 1, 1)
        plt.title(type1)
        ax.set_ylim([0, 260])
        ax.violinplot(df2.iloc[:, 5:11].values.T.tolist())
        ax.set_xticks([1, 2, 3, 4, 5, 6]) #データ範囲のどこに目盛りが入るかを指定する
        ax.set_xticklabels(df2.columns[5:11], rotation=90)
        plt.grid()
        plt.show()


with code():
    from sklearn.decomposition import PCA #主成分分析器


with code():
    #主成分分析の実行
    pca = PCA()
    pca.fit(df.iloc[:, 5:11])
    # データを主成分空間に写像 = 次元圧縮
    feature = pca.transform(df.iloc[:, 5:11])
    # 第一主成分と第二主成分でプロットする
    plt.figure(figsize=(8, 8))
    plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid()
    plt.show()


with code():
    # 累積寄与率を図示する
    import matplotlib.ticker as ticker
    import numpy as np
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative contribution ratio")
    plt.grid()
    plt.show()


with code():
    #主成分分析の実行
    pca = PCA()
    pca.fit(df.iloc[:, 5:11])
    # データを主成分空間に写像 = 次元圧縮
    feature = pca.transform(df.iloc[:, 5:11])
    # 第一主成分と第二主成分でプロットする
    plt.figure(figsize=(8, 8))
    for type1 in type1s:
        plt.scatter(feature[df['Type 1'] == type1, 0], feature[df['Type 1'] == type1, 1], alpha=0.8, label=type1)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc = 'upper right',
              bbox_to_anchor = (0.7, 0.7, 0.5, 0.1),
              borderaxespad = 0.0)
    plt.grid()
    plt.show()


with code():
    #主成分分析の実行
    pca = PCA()
    pca.fit(df.iloc[:, 5:11])
    # データを主成分空間に写像 = 次元圧縮
    feature = pca.transform(df.iloc[:, 5:11])
    # 第一主成分と第二主成分でプロットする
    plt.figure(figsize=(8, 8))
    for generation in range(0, 7):
        plt.scatter(feature[df['Generation'] == generation, 0], feature[df['Generation'] == generation, 1], alpha=0.8, label=generation)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc = 'upper right',
              bbox_to_anchor = (0.7, 0.7, 0.5, 0.1),
              borderaxespad = 0.0)
    plt.grid()
    plt.show()


with code():
    #主成分分析の実行
    pca = PCA()
    pca.fit(df.iloc[:, 5:11])
    # データを主成分空間に写像 = 次元圧縮
    feature = pca.transform(df.iloc[:, 5:11])
    # 第一主成分と第二主成分でプロットする
    plt.figure(figsize=(8, 8))
    for binary in [True, False]:
        plt.scatter(feature[df['Legendary'] == binary, 0], feature[df['Legendary'] == binary, 1], alpha=0.8, label=binary)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc = 'upper right',
              bbox_to_anchor = (0.7, 0.7, 0.5, 0.1),
              borderaxespad = 0.0)
    plt.grid()
    plt.show()


with code():
    # 第一主成分と第二主成分における観測変数の寄与度をプロットする
    plt.figure(figsize=(8, 8))
    for x, y, name in zip(pca.components_[0], pca.components_[1], df.columns[5:11]):
        plt.text(x, y, name)
    plt.scatter(pca.components_[0], pca.components_[1])
    plt.grid()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


with code():
    from sklearn.decomposition import FactorAnalysis
    fa = FactorAnalysis(n_components=2, max_iter=500)
    factors = fa.fit_transform(df.iloc[:, 5:11])


with code():
    plt.figure(figsize=(8, 8))
    for binary in [True, False]:
        plt.scatter(factors[df['Legendary'] == binary, 0], factors[df['Legendary'] == binary, 1], alpha=0.8, label=binary)
    plt.xlabel("Factor 1")
    plt.ylabel("Factor 2")
    plt.legend(loc = 'upper right',
              bbox_to_anchor = (0.7, 0.7, 0.5, 0.1),
              borderaxespad = 0.0)
    plt.grid()
    plt.show()


with code():
    # 第一主成分と第二主成分における観測変数の寄与度をプロットする
    plt.figure(figsize=(8, 8))
    for x, y, name in zip(fa.components_[0], fa.components_[1], df.columns[5:11]):
        plt.text(x, y, name)
    plt.scatter(fa.components_[0], fa.components_[1])
    plt.grid()
    plt.xlabel("Factor 1")
    plt.ylabel("Factor 2")
    plt.show()


with code():
    # 行列の正規化
    dfs = df.iloc[:, 5:11].apply(lambda x: (x-x.mean())/x.std(), axis=0)


with code():
    # metric は色々あるので、ケースバイケースでどれかひとつ好きなものを選ぶ。
    # method も色々あるので、ケースバイケースでどれかひとつ好きなものを選ぶ。
    from scipy.cluster.hierarchy import linkage, dendrogram
    result1 = linkage(dfs,
                      #metric = 'braycurtis',
                      #metric = 'canberra',
                      #metric = 'chebyshev',
                      #metric = 'cityblock',
                      #metric = 'correlation',
                      #metric = 'cosine',
                      metric = 'euclidean',
                      #metric = 'hamming',
                      #metric = 'jaccard',
                      #method= 'single')
                      method = 'average')
                      #method= 'complete')
                      #method='weighted')
    plt.figure(figsize=(8, 128))
    dendrogram(result1, orientation='right', labels=list(df['Name']), color_threshold=2)
    plt.title("Dedrogram")
    plt.xlabel("Threshold")
    plt.grid()
    plt.show()


with code():
    # 指定したクラスタ数でクラスタを得る関数を作る。
    def get_cluster_by_number(result, number):
        output_clusters = []
        x_result, y_result = result.shape
        n_clusters = x_result + 1
        cluster_id = x_result + 1
        father_of = {}
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        for i in range(len(result) - 1):
            n1 = int(result[i][0])
            n2 = int(result[i][1])
            val = result[i][2]
            n_clusters -= 1
            if n_clusters >= number:
                father_of[n1] = cluster_id
                father_of[n2] = cluster_id

            cluster_id += 1

        cluster_dict = {}
        for n in range(x_result + 1):
            if n not in father_of:
                output_clusters.append([n])
                continue

            n2 = n
            m = False
            while n2 in father_of:
                m = father_of[n2]
                #print [n2, m]
                n2 = m

            if m not in cluster_dict:
                cluster_dict.update({m:[]})
            cluster_dict[m].append(n)

        output_clusters += cluster_dict.values()

        output_cluster_id = 0
        output_cluster_ids = [0] * (x_result + 1)
        for cluster in sorted(output_clusters):
            for i in cluster:
                output_cluster_ids[i] = output_cluster_id
            output_cluster_id += 1

        return output_cluster_ids


with code():
    clusterIDs = get_cluster_by_number(result1, 50)
    print(clusterIDs)


with code():
    plt.hist(clusterIDs, bins=50)
    plt.grid()
    plt.show()


with code():
    for i in range(max(clusterIDs) + 1):
        cluster = []
        for j, k in enumerate(clusterIDs):
            if i == k:
                cluster.append(j)
        fig = plt.figure()
        print("Cluster {}: {} samples".format(i + 1, len(cluster)))
        for j in cluster:
            labels = list(df.columns[5:11])
            values = list(df.iloc[j, 5:11])
            angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)
            values = np.concatenate((values, [values[0]]))  # 閉じた多角形にする
            ax = fig.add_subplot(111, polar=True)
            ax.plot(angles, values, 'o-', label=df.iloc[j, :]['Name'] + " (" + df.iloc[j, :]['Type 1'] + ")")  # 外枠
            #ax.fill(angles, values, alpha=0.25)  # 塗りつぶし
            ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)  # 軸ラベル
            ax.set_rlim(0 ,250)
        plt.legend( loc = 'center right',
              bbox_to_anchor = (1.5, 0.5, 0.5, 0.1),
              borderaxespad = 0.0)
        plt.show()


with code():
    X = df.iloc[:, 5:11]
    y = df['Total']


with code():
    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    regr.fit(X, y) # 予測モデルを作成

    print("回帰係数= ", regr.coef_)
    print("切片= ", regr.intercept_)
    print("決定係数= ", regr.score(X, y))


with code():
    df.head()


with code():
    df.columns[[5, 6, 7, 10]]


with code():
    X = df.iloc[:, [5, 6, 7, 10]]
    y = df['Sp. Atk']


with code():
    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    regr.fit(X, y) # 予測モデルを作成

    print("回帰係数= ", regr.coef_)
    print("切片= ", regr.intercept_)
    print("決定係数= ", regr.score(X, y))


with code():
    #from sklearn.cross_validation import train_test_split # 訓練データとテストデータに分割
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4) # 訓練データ・テストデータへのランダムな分割


with code():
    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train) # 予測モデルを作成

    print("回帰係数= ", regr.coef_)
    print("切片= ", regr.intercept_)
    print("決定係数(train)= ", regr.score(X_train, y_train))
    print("決定係数(test)= ", regr.score(X_test, y_test))


with code():
    Xs = X.apply(lambda x: (x-x.mean())/x.std(), axis=0)
    ys = list(pd.DataFrame(y).apply(lambda x: (x-x.mean())/x.std()).values.reshape(len(y),))


with code():
    from sklearn import linear_model
    regr = linear_model.LinearRegression()

    regr.fit(Xs, ys) # 予測モデルを作成

    print("標準回帰係数= ", regr.coef_)
    print("切片= ", regr.intercept_)
    print("決定係数= ", regr.score(Xs, ys))


with code():
    pd.DataFrame(regr.coef_, index=list(df.columns[[5, 6, 7, 10]])).sort_values(0, ascending=False).style.bar(subset=[0])


with code():
    df2 = pd.get_dummies(df.iloc[:, 2:], dummy_na=True)
    df2.head()


with code():
    X = df2
    del X['Total']
    del X['Sp. Atk']
    del X['Sp. Def']
    del X['Legendary']
    X.head()


with code():
    #from sklearn.cross_validation import train_test_split # 訓練データとテストデータに分割
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4) # 訓練データ・テストデータへのランダムな分割


with code():
    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train) # 予測モデルを作成

    print("回帰係数= ", regr.coef_)
    print("切片= ", regr.intercept_)
    print("決定係数(train)= ", regr.score(X_train, y_train))
    print("決定係数(test)= ", regr.score(X_test, y_test))


with code():
    from sklearn.ensemble import RandomForestRegressor
    regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
    regr.fit(X_train, y_train)
    print("決定係数(train)= ", regr.score(X_train, y_train))
    print("決定係数(test)= ", regr.score(X_test, y_test))


with code():
    regr.feature_importances_


with code():
    pd.DataFrame(regr.feature_importances_, index=list(X.columns)).sort_values(0, ascending=False).head(10).style.bar(subset=[0])


with code():
    df2 = pd.get_dummies(df.iloc[:, 2:], dummy_na=True)
    X = df2
    del X['Total']
    del X['Sp. Atk']
    del X['Sp. Def']
    del X['Legendary']
    X.head()
    y = df['Legendary']


with code():
    #from sklearn.cross_validation import train_test_split # 訓練データとテストデータに分割
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4) # 訓練データ・テストデータへのランダムな分割


with code():
    from sklearn.linear_model import LogisticRegression # ロジスティック回帰
    clf = LogisticRegression(solver='lbfgs', max_iter=10000) #モデルの生成
    clf.fit(X_train, y_train) #学習
    print("正解率(train): ", clf.score(X_train,y_train))
    print("正解率(test): ", clf.score(X_test,y_test))


with code():
    from sklearn.metrics import confusion_matrix # 混合行列
    # 予測結果と、正解（本当の答え）がどのくらい合っていたかを表す混合行列
    pd.DataFrame(confusion_matrix(clf.predict(X_test), y_test), index=['predicted 0', 'predicted 1'], columns=['real 0', 'real 1'])


with code():
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(max_iter=10000)
    clf.fit(X_train, y_train) #学習
    print("正解率(train): ", clf.score(X_train,y_train))
    print("正解率(test): ", clf.score(X_test,y_test))


with code():
    from sklearn.metrics import confusion_matrix # 混合行列
    # 予測結果と、正解（本当の答え）がどのくらい合っていたかを表す混合行列
    pd.DataFrame(confusion_matrix(clf.predict(X_test), y_test), index=['predicted 0', 'predicted 1'], columns=['real 0', 'real 1'])


with code():
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train) #学習
    print("正解率(train): ", clf.score(X_train,y_train))
    print("正解率(test): ", clf.score(X_test,y_test))


with code():
    from sklearn.metrics import confusion_matrix # 混合行列
    # 予測結果と、正解（本当の答え）がどのくらい合っていたかを表す混合行列
    pd.DataFrame(confusion_matrix(clf.predict(X_test), y_test), index=['predicted 0', 'predicted 1'], columns=['real 0', 'real 1'])


with code():
    pd.DataFrame(clf.feature_importances_, index=list(X.columns)).sort_values(0, ascending=False).style.bar(subset=[0])


with code():
    from sklearn.metrics import roc_curve, precision_recall_curve, auc, classification_report, confusion_matrix
    # AUCスコアを出す。
    probas = clf.fit(X_train, y_train).predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
    roc_auc = auc(fpr, tpr)
    print ("ROC score : ",  roc_auc)


with code():
    # ROC curve を描く
    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


with code():
    # AUPRスコアを出す
    precision, recall, thresholds = precision_recall_curve(y_test, probas[:, 1])
    area = auc(recall, precision)
    print ("AUPR score: " , area)


with code():
    # PR curve を描く
    plt.clf()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example: AUPR=%0.2f' % area)
    plt.legend(loc="lower left")
    plt.show()
