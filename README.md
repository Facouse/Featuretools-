# Featuretools
# Featuretools介绍
人工特性工程是一项冗长乏味的任务，并且受到人类想象力的限制——我们可以思考创建的特性只有这么多，而且随着时间的推移，创建新特性需要大量的时间。理想情况下，应该有一个客观的方法来创建一系列不同的候选新特性，然后我们可以将这些特性用于机器学习任务。这个过程的目的不是替换数据科学家，而是使他的工作更容易，并允许他使用自动工作流补充领域知识。

![Featuretools](https://img-blog.csdnimg.cn/4a3ae08c40d341dcb55056d94d2ed84d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBARmFjb3VzZQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
# Featuretools快速开始
以官方文档为例：

下面是使用深度特征合成 (DFS) 执行自动化特征工程的示例。在此示例中，我们将 DFS 应用于由带有时间戳的客户交易组成的多表数据集。

```python
import featuretools as ft #引用featuretools库
```
**加载模拟数据**

```python
data = ft.demo.load_mock_customer()
```
**准备数据**
在这个模拟数据集中，有 3 个 DataFrame：

 - customers: unique customers who had sessions
 - sessions: unique sessions and associated attributes
 - transactions: list of events in this session

*注意我标红的地方*

```python
customers_df = data["customers"]
customers_df
```
![customers](https://img-blog.csdnimg.cn/64730dcbe17547068ed6967791f38d09.png)
```python
sessions_df = data["sessions"]
sessions_df.sample(5)
```
![sessions](https://img-blog.csdnimg.cn/49254e27313c411f8a07102b6576b173.png)

```python
transactions_df = data["transactions"]
transactions_df.sample(5)
```
![transactions](https://img-blog.csdnimg.cn/9941a9dc14d44b898e94ea1d3469c3ab.png)
首先，我们指定一个包含数据集中所有 DataFrame 的字典。如果 DataFrame 存在索引列和时间索引列，则 DataFrame 将与其索引列和时间索引列一起传入。

```python
dataframes = {
   "customers" : (customers_df, "customer_id"),
   "sessions" : (sessions_df, "session_id", "session_start"),
   "transactions" : (transactions_df, "transaction_id", "transaction_time")
}
```
其次，我们指定 DataFrames 是如何关联的。当两个 DataFrame 具有一对多关系时，我们称“一个”DataFrame，即“父 DataFrame”。sessions中包含重复的customer_id值，即customers是父而sessions是子，父子关系定义如下：

```python
(parent_dataframe, parent_column, child_dataframe, child_column)
```
在这个数据集中，我们有两个关系

```python
relationships = [("sessions", "session_id", "transactions", "session_id"),
                 ("customers", "customer_id", "sessions", "customer_id")] #左为父，右为子
```
**运行深度特征合成**
DFS 的最小输入是 DataFrame 的字典（dataframes）、关系列表（relationships）以及我们要计算其特征的目标 DataFrame 的名称（target_dataframe_name）。 DFS 的输出是一个特征矩阵和相应的特征定义列表。

```python
feature_matrix_customers, features_defs = ft.dfs(dataframes=dataframes,
                                                 relationships=relationships,
                                                 target_dataframe_name="customers")
feature_matrix_customers
```
![输出结果](https://img-blog.csdnimg.cn/465cc887aa624258bd67c5581847d874.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBARmFjb3VzZQ==,size_20,color_FFFFFF,t_70,g_se,x_16)
我们现在有几十个新特征来描述客户的行为。
**更改目标 DataFrame**
DFS 如此强大的原因之一是它可以为我们的 EntitySet 中的任何 DataFrame 创建一个特征矩阵。例如，如果我们想为会话构建功能。

```python
feature_matrix_sessions, features_defs = ft.dfs(dataframes=dataframes,
                                                relationships=relationships,
                                                target_dataframe_name="sessions")
feature_matrix_sessions.head(5)
```
![输出结果](https://img-blog.csdnimg.cn/8087fc0029aa40f7950905d2e98bc2f1.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBARmFjb3VzZQ==,size_20,color_FFFFFF,t_70,g_se,x_16)
**了解特征输出**
一般来说，Featuretools 通过特征名称引用生成的特征。为了让特性更容易理解，Featuretools 提供了两个额外的工具，featuretools.graph_feature() 和 featuretools.describe_feature()，帮助解释什么是特征以及 Featuretools 生成它的步骤。让我们看一下这个示例功能。

```python
feature = features_defs[18]
feature
```
![特征描述](https://img-blog.csdnimg.cn/2969d6d69d5d46dda7d7b64643904fd6.png)
特征谱系图
特征谱系图直观地遍历特征生成。从基础数据开始，它们逐步显示应用的基元和生成的中间特征以创建最终特征。

```python
ft.graph_feature(feature)
```
![特征谱系图](https://img-blog.csdnimg.cn/430a915a2fcc4dbc920abc5aa92241f7.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBARmFjb3VzZQ==,size_20,color_FFFFFF,t_70,g_se,x_16)
功能描述
Featuretools 还可以自动生成特征的英文句子描述。功能描述有助于解释什么是功能，并且可以通过包含手动定义的自定义定义来进一步改进。关如何自定义自动生成的功能描述的更多详细信息，请参阅生成功能描述。

```python
ft.describe_feature(feature)
```
'The most frequently occurring value of the year of the "transaction_time" of all instances of "transactions" for each "session_id" in "sessions".'

这样，我们就基于三个DataFrame利用Featuretools生成十几个特征，供我们挖掘数据信息。后面会利用用实体集来表示数据，而不是字典。
