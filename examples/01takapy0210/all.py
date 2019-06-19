from nbreversible import code
with code():
    import matplotlib.pyplot as plt #Visulization
    import seaborn as sns #Visulization
    #%matplotlib inline



with code():
    plt.hist(train['quality'], bins=12)
    plt.title("quality_Histgram")
    plt.xlabel('quality')
    plt.ylabel('count')
    plt.show()


with code():
    random.seed(0)
    plt.figure(figsize=(20, 6))
    plt.hist(np.random.randn(10**5)*10 + 50, bins=60,range=(20,80))
    plt.grid(True)


with code():
    plt.figure(figsize=(4,3),facecolor="white")

    Y1 = np.array([30,10,40])
    Y2 = np.array([10,50,90])

    X = np.arange(len(Y1))
    # X = [0 1 2]

    # グラフの幅
    w=0.4

    # グラフの大きさ指定
    plt.figure(figsize=(15, 6))

    plt.bar(X, Y1, color='b', width=w, label='Math first', align="center")
    plt.bar(X + w, Y2, color='g', width=w, label='Math final', align="center")

    # 凡例を最適な位置に配置
    plt.legend(loc="best")

    plt.xticks(X + w/2, ['Class A','Class B','Class C'])
    plt.grid(True)


with code():
    left = np.array([1, 2, 3, 4, 5])

    height1 = np.array([100, 200, 300, 400, 500])
    height2 = np.array([1000, 800, 600, 400, 200])

    # グラフの大きさ指定
    plt.figure(figsize=(15, 6))

    p1 = plt.bar(left, height1, color="blue")
    p2 = plt.bar(left, height2, bottom=height1, color="lightblue")

    plt.legend((p1[0], p2[0]), ("Class 1", "Class 2"))


with code():
    labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
    sizes = [15, 30, 45, 10]
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    explode = (0, 0.1, 0, 0)  # 円から切り離して表示させることが可能

    # グラフの大きさ指定
    plt.figure(figsize=(15, 6))

    # startangleは各要素の出力を開始する角度を表す(反時計回りが正), 向きはcounterclockで指定可能
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')


with code():
    N = 25

    # X,Y軸
    x = np.random.rand(N)
    y = np.random.rand(N)

    # color番号
    colors = np.random.rand(N)

    # バブルの大きさ
    area = 10 * np.pi * (15 * np.random.rand(N))**2

    # グラフの大きさ指定
    plt.figure(figsize=(15, 6))

    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.grid(True)


with code():
    #　シード値の固定
    random.seed(0)
    x = np.random.randn(50) # x軸のデータ
    y = np.sin(x) + np.random.randn(50) # y軸のデータ
    plt.figure(figsize=(16, 6)) # グラフの大きさ指定

    # グラフの描写
    plt.plot(x, y, "o")

    #以下でも散布図が描ける
    #plt.scatter(x, y)

    # title
    plt.title("Title Name")
    # Xの座標名
    plt.xlabel("X")
    # Yの座標名
    plt.ylabel("Y")

    # gridの表示
    plt.grid(True)


with code():
    # シード値の指定
    np.random.seed(0)
    # データの範囲
    numpy_data_x = np.arange(1000)

    # 乱数の発生と積み上げ
    numpy_random_data_y = np.random.randn(1000).cumsum()

    # グラフの大きさを指定
    plt.figure(figsize=(20, 6))

    # label=とlegendでラベルをつけることが可能
    plt.plot(numpy_data_x,numpy_random_data_y,label="Label")
    plt.legend()

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)


with code():
    # 関数の定義
    def sample_function(x):
        return (x**2 + 2*x + 1)

    x = np.arange(-10, 10)
    plt.figure(figsize=(20, 6))
    plt.plot(x, sample_function(x))
    plt.grid(True)


with code():
    k = 13 # 表示する特徴量の数
    corrmat = train.corr()
    cols = corrmat.nlargest(k, 'quality').index # リストの最大値から順にk個の要素の添字(index)を取得
    # df_train[cols].head()
    cm = np.corrcoef(train[cols].values.T) # 相関関数行列を求める ※転置が必要
    sns.set(font_scale=1.25)
    f, ax = plt.subplots(figsize=(16, 12))
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()


with code():
    sns.set()
    cols = ['quality', 'alcohol', 'citric.acid', 'free.sulfur.dioxide', 'sulphates', 'pH', 'total.sulfur.dioxide'] # プロットしたい特徴量
    sns.pairplot(train[cols], size = 2.0)
    plt.show()


with code():
    sns.jointplot(x="quality", y="alcohol", data=train, ratio=3, color="r", size=6)
    plt.show()


with code():
    def plot_feature_importance(model):
      n_features = X.shape[1]
      plt.barh(range(n_features), model.feature_importances_, align='center')
      plt.yticks(np.arange(n_features), X.columns)
      plt.xlabel('Feature importance')
      plt.ylabel('Feature')

    # ※Xはtrain_test_splitで分割する前のtrainデータを想定


with code():
    # ランダムフォレスト
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=100, random_state=20181101) # n_estimatorsは構築する決定木の数
    forest.fit(X_train, y_train)

    # 表示
    plot_feature_importance(forest)


with code():
    import lightgbm as lgb
    # 可視化（modelはlightgbmで学習させたモデル）
    lgb.plot_importance(model, figsize=(12, 8))
    plt.show()
