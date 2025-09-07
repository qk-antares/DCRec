import pandas as pd
import numpy as np
import os

def preprocess_ml100k():
    """
    预处理MovieLens-100K数据集
    """
    # 数据路径
    data_dir = "data/ml-100k"
    output_dir = "data/ml-100k/processed"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print("开始加载ML-100K数据...")
    
    # 1. 加载u.data (ratings数据)
    print("加载u.data...")
    ratings_df = pd.read_csv(f"{data_dir}/u.data", sep='\t', 
                            names=['user_id', 'item_id', 'rating', 'timestamp'])
    print(f"加载了 {len(ratings_df)} 条评分数据")
    
    # 2. 加载u.user (用户数据)
    print("加载u.user...")
    users_df = pd.read_csv(f"{data_dir}/u.user", sep='|', 
                          names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
    print(f"加载了 {len(users_df)} 个用户数据")
    
    # 3. 加载u.item (物品数据)
    print("加载u.item...")
    # u.item有24列：item_id, movie_title, release_date, video_release_date, IMDb_URL + 19个genre列
    genre_names = ['unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 
                   'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                   'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    
    column_names = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'] + genre_names
    
    movies_df = pd.read_csv(f"{data_dir}/u.item", sep='|', names=column_names, encoding='latin-1')
    print(f"加载了 {len(movies_df)} 个电影数据")
    
    print("数据加载完成，开始预处理...")
    
    # Step 1: 对ratings按timestamp排序
    ratings_df = ratings_df.sort_values('timestamp').reset_index(drop=True)
    print("Step 1: 按时间戳排序完成")
    
    # Step 2: 将timestamp转换为相对偏移量
    min_timestamp = ratings_df['timestamp'].min()
    ratings_df['ts'] = ratings_df['timestamp'] - min_timestamp
    print("Step 2: 时间戳转换完成")
    
    # Step 3: 只保留正样本（评分>=4），删除其他样本
    ratings_df = ratings_df[ratings_df['rating'] >= 4].reset_index(drop=True)
    ratings_df['label'] = 1  # 所有保留的样本都是正样本
    print(f"Step 3: 只保留正样本，剩余 {len(ratings_df)} 条记录")
    
    # Step 4: 重新编码user_id和item_id
    # 按照排序后的数据，按行读取，遇到的第一个user_id编码为1，第二个不同的user_id编码为2，以此类推
    user_id_map = {}
    item_id_map = {}
    user_counter = 1
    
    # 首先遍历所有数据，为用户建立映射
    for _, row in ratings_df.iterrows():
        user_id = row['user_id']
        if user_id not in user_id_map:
            user_id_map[user_id] = user_counter
            user_counter += 1
    
    # 然后为物品建立映射，起始编码为user_id_count+1
    item_counter = user_counter  # 即len(user_id_map) + 1
    for _, row in ratings_df.iterrows():
        item_id = row['item_id']
        if item_id not in item_id_map:
            item_id_map[item_id] = item_counter
            item_counter += 1
    
    print(f"唯一用户数: {len(user_id_map)}, 唯一物品数: {len(item_id_map)}")
    print(f"用户ID编码范围: 1-{len(user_id_map)}")
    print(f"物品ID编码范围: {len(user_id_map)+1}-{len(user_id_map)+len(item_id_map)}")
    
    # 应用映射
    ratings_df['u'] = ratings_df['user_id'].map(user_id_map)
    ratings_df['i'] = ratings_df['item_id'].map(item_id_map)
    print("Step 4: ID重新编码完成")
    
    # Step 5: 为每个交互编码idx
    ratings_df['idx'] = range(1, len(ratings_df) + 1)
    print("Step 5: 交互编码完成")
    
    # 生成graph.csv
    graph_df = ratings_df[['idx', 'u', 'i', 'ts', 'label']]
    graph_df.to_csv(f"{output_dir}/graph.csv", index=False)
    print(f"graph.csv已保存，包含 {len(graph_df)} 条记录")
    
    # 生成id_mapping.csv
    id_mapping_data = []
    # 添加用户映射
    for old_id, new_id in user_id_map.items():
        id_mapping_data.append([new_id, f"u_{old_id}"])
    # 添加物品映射
    for old_id, new_id in item_id_map.items():
        id_mapping_data.append([new_id, f"i_{old_id}"])
    
    id_mapping_df = pd.DataFrame(id_mapping_data, columns=['id', 'original_id'])
    id_mapping_df = id_mapping_df.sort_values('id')
    id_mapping_df.to_csv(f"{output_dir}/id_mapping.csv", index=False)
    print(f"id_mapping.csv已保存，包含 {len(id_mapping_df)} 条映射")
    
    # 生成节点特征 (nodes.npy)
    print("开始生成节点特征...")
    
    # 统计用户特征的所有可能值
    print("统计用户特征...")
    unique_ages = sorted(users_df['age'].unique())
    unique_genders = sorted(users_df['gender'].unique())
    unique_occupations = sorted(users_df['occupation'].unique())
    
    print(f"原始年龄类型: {len(unique_ages)} 种，范围: {min(unique_ages)}-{max(unique_ages)}")
    print(f"性别类型: {unique_genders} (数量: {len(unique_genders)})")
    print(f"职业类型: {len(unique_occupations)} 种")
    
    # 定义年龄段映射函数
    def map_age_to_group(age):
        """将年龄映射到年龄段"""
        if age < 18:
            return "Under 18"
        elif age <= 24:
            return "18-24"
        elif age <= 34:
            return "25-34"
        elif age <= 44:
            return "35-44"
        elif age <= 49:
            return "45-49"
        elif age <= 55:
            return "50-55"
        else:
            return "56+"
    
    # 应用年龄段映射
    users_df['age_group'] = users_df['age'].apply(map_age_to_group)
    unique_age_groups = ["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
    
    print(f"压缩后年龄段: {unique_age_groups} (数量: {len(unique_age_groups)})")
    
    # 统计每个年龄段的用户数量
    age_group_counts = users_df['age_group'].value_counts()
    print("年龄段分布:")
    for age_group in unique_age_groups:
        count = age_group_counts.get(age_group, 0)
        print(f"  {age_group}: {count} 用户")
    
    # 创建用户特征映射
    age_group_to_idx = {age_group: idx for idx, age_group in enumerate(unique_age_groups)}
    gender_to_idx = {gender: idx for idx, gender in enumerate(unique_genders)}
    occupation_to_idx = {occupation: idx for idx, occupation in enumerate(unique_occupations)}
    
    # 统计物品特征 - 类型(genres)和年份
    print("统计物品特征...")
    # 获取所有的genre列名（除了'unknown'）
    active_genres = [genre for genre in genre_names if genre != 'unknown']
    print(f"电影类型: {active_genres} (数量: {len(active_genres)})")
    
    # 提取电影年份
    print("提取电影年份...")
    movies_df['year'] = movies_df['movie_title'].str.extract(r'\((\d{4})\)')[0]
    
    # 处理缺失年份的情况
    missing_years = movies_df['year'].isna().sum()
    if missing_years > 0:
        print(f"发现 {missing_years} 部电影缺少年份信息，将使用默认年份1996")
        movies_df['year'] = movies_df['year'].fillna('1996')
    
    movies_df['year'] = movies_df['year'].astype(int)
    unique_years = sorted(movies_df['year'].unique())
    print(f"电影年份范围: {min(unique_years)}-{max(unique_years)} (数量: {len(unique_years)})")
    
    # 创建年份特征映射
    year_to_idx = {year: idx for idx, year in enumerate(unique_years)}
    
    # 计算特征维度
    user_feature_dim = len(unique_age_groups) + len(unique_genders) + len(unique_occupations)
    item_feature_dim = len(unique_years) + len(active_genres)  # 年份 + genre类型
    total_nodes = len(user_id_map) + len(item_id_map) + 1  # +1 for index 0
    feature_dim = user_feature_dim + item_feature_dim
    
    print(f"用户特征维度: {user_feature_dim} (年龄段:{len(unique_age_groups)} + 性别:{len(unique_genders)} + 职业:{len(unique_occupations)})")
    print(f"物品特征维度: {item_feature_dim} (年份:{len(unique_years)} + 类型:{len(active_genres)})")
    print(f"总特征维度: {feature_dim}")
    print(f"总节点数: {total_nodes}")
    
    # 初始化节点特征矩阵
    nodes_features = np.zeros((total_nodes, feature_dim), dtype=np.float32)
    
    # 填充用户特征
    print("填充用户特征...")
    for old_user_id in user_id_map.keys():
        new_user_id = user_id_map[old_user_id]
        user_info = users_df[users_df['user_id'] == old_user_id].iloc[0]
        
        feature_vector = np.zeros(feature_dim, dtype=np.float32)
        offset = 0
        
        # Age Group特征 (独热编码)
        age_group = map_age_to_group(user_info['age'])
        age_group_idx = age_group_to_idx[age_group]
        feature_vector[offset + age_group_idx] = 1
        offset += len(unique_age_groups)
        
        # Gender特征 (独热编码)
        gender_idx = gender_to_idx[user_info['gender']]
        feature_vector[offset + gender_idx] = 1
        offset += len(unique_genders)
        
        # Occupation特征 (独热编码)
        occupation_idx = occupation_to_idx[user_info['occupation']]
        feature_vector[offset + occupation_idx] = 1
        
        nodes_features[new_user_id] = feature_vector
    
    # 填充物品特征
    print("填充物品特征...")
    for old_item_id in item_id_map.keys():
        new_item_id = item_id_map[old_item_id]
        movie_info = movies_df[movies_df['item_id'] == old_item_id].iloc[0]
        
        feature_vector = np.zeros(feature_dim, dtype=np.float32)
        
        # 物品特征在用户特征之后
        offset = user_feature_dim
        
        # Year特征 (独热编码)
        year_idx = year_to_idx[movie_info['year']]
        feature_vector[offset + year_idx] = 1
        offset += len(unique_years)
        
        # Genres特征 (多标签，可以同时属于多个类型)
        for i, genre in enumerate(active_genres):
            if movie_info[genre] == 1:  # 如果属于该类型
                feature_vector[offset + i] = 1
        
        nodes_features[new_item_id] = feature_vector
    
    # 保存节点特征
    np.save(f"{output_dir}/nodes.npy", nodes_features)
    print(f"nodes.npy已保存，形状: {nodes_features.shape}")
    
    # 生成边特征 (edges.npy)
    print("开始生成边特征...")
    
    # 创建5维的边特征向量，对应1-5的评分，首行为全0
    edges_features = np.zeros((len(ratings_df)+1, 5), dtype=np.float32)
    
    # 按ratings_df['idx']赋值
    for row in ratings_df.itertuples():
        edges_features[row.idx, row.rating - 1] = 1  # idx从1开始，rating从1开始所以减1
    
    # 保存边特征
    np.save(f"{output_dir}/edges.npy", edges_features)
    print(f"edges.npy已保存，形状: {edges_features.shape}")
    
    # 打印一些统计信息
    print("\n=== 数据集统计信息 ===")
    original_ratings = pd.read_csv(f'{data_dir}/u.data', sep='\t', names=['u','i','r','t'])
    print(f"原始评分数据: {len(original_ratings)} 条")
    print(f"过滤后评分数据: {len(ratings_df)} 条")
    print(f"用户数: {len(user_id_map)}")
    print(f"物品数: {len(item_id_map)}")
    print(f"总节点数: {total_nodes}")
    print(f"节点特征维度: {feature_dim}")
    print(f"边特征维度: 5")
    
    print("\n预处理完成！")
    print(f"输出文件保存在: {output_dir}")
    print("生成的文件:")
    print("- graph.csv: 图结构数据")
    print("- id_mapping.csv: ID映射关系")
    print("- nodes.npy: 节点特征")
    print("- edges.npy: 边特征")

if __name__ == "__main__":
    preprocess_ml100k()
