import json
import datetime
from pyecharts.charts import Pie
from pyecharts import options as opts

# 读原始数据文件
today = datetime.date.today().strftime('%Y%m%d')   #20200315
datafile = 'data/'+ today + '.json'
with open(datafile, 'r', encoding='UTF-8') as file:
    json_array = json.loads(file.read())

# 分析全国实时确诊数据：'confirmedCount'字段
china_data = []
for province in json_array:
    china_data.append((province['provinceShortName'], province['confirmedCount']))
china_data = sorted(china_data, key=lambda x: x[1], reverse=True)                 #reverse=True,表示降序，反之升序


print(china_data)

# 开始作图
pie = Pie()
labels = [data[0] for data in china_data]
counts = [data[1] for data in china_data]

pie.add("累计确诊", [list(z) for z in zip(labels, counts)])
pie.set_series_opts(label_opts=opts.LabelOpts(font_size=12, formatter="{b}: {c}"), is_show=False)
pie.set_global_opts(title_opts=opts.TitleOpts(title="全国累计确诊饼状图", subtitle='数据来源：丁香园'), legend_opts=opts.LegendOpts(is_show=False))

pie.render(path='./data/全国实时确诊数据饼状图.html')