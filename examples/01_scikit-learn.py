import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
import sklearn.neighbors

assert sklearn.__version__ >= "0.20"
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    oecd_bli = oecd_bli.pivot(
        index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

# 加载数据
path = "../datasets/lifesat/"
oecd_bli = pd.read_csv(path+"oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv(path+"gdp_per_capita.csv", thousands=',', delimiter='\t',
                             encoding='latin1', na_values="n/a")


# 准备数据
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# 可视化数据
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()

# 选择线性模型
# lin_reg_model = sklearn.linear_model.LinearRegression()
lin_reg_model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)

# 训练模型
lin_reg_model.fit(X, y)
print("trainning ended")
# 对塞浦路斯进行预测
X_new = [[22587]]  # 塞浦路斯的人均 GDP
print(lin_reg_model.predict(X_new))  # outputs [[ 5.96242338]]
