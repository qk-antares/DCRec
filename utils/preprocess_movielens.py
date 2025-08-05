import pandas as pd
import numpy as np
import os

def preprocess_movielens():
    """
    预处理MovieLens-1M数据集
    """
    # 数据路径
    data_dir = "data/ml-1m"
    output_dir = "data/ml-1m/processed"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print("开始加载数据...")
    
    # 1. 加载ratings数据
    ratings_data = []
    with open(f"{data_dir}/ratings.dat", "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            user_id, item_id, rating, timestamp = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            ratings_data.append([user_id, item_id, rating, timestamp])
    
    ratings_df = pd.DataFrame(ratings_data, columns=['user_id', 'item_id', 'rating', 'timestamp'])
    print(f"加载了 {len(ratings_df)} 条评分数据")
    
    # 2. 加载users数据
    users_data = []
    with open(f"{data_dir}/users.dat", "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            user_id, gender, age, occupation, zip_code = int(parts[0]), parts[1], int(parts[2]), int(parts[3]), parts[4]
            users_data.append([user_id, gender, age, occupation, zip_code])
    
    users_df = pd.DataFrame(users_data, columns=['user_id', 'gender', 'age', 'occupation', 'zip_code'])
    print(f"加载了 {len(users_df)} 个用户数据")
    
    # 3. 加载movies数据
    movies_data = []
    with open(f"{data_dir}/movies.dat", "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            item_id, title, genres = int(parts[0]), parts[1], parts[2]
            movies_data.append([item_id, title, genres])
    
    movies_df = pd.DataFrame(movies_data, columns=['item_id', 'title', 'genres'])
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
    unique_genders = sorted(users_df['gender'].unique())
    unique_ages = sorted(users_df['age'].unique())
    unique_occupations = sorted(users_df['occupation'].unique())
    
    print(f"性别类型: {unique_genders} (数量: {len(unique_genders)})")
    print(f"年龄类型: {unique_ages} (数量: {len(unique_ages)})")
    print(f"职业类型: {unique_occupations} (数量: {len(unique_occupations)})")
    
    # 创建用户特征映射
    gender_to_idx = {gender: idx for idx, gender in enumerate(unique_genders)}
    age_to_idx = {age: idx for idx, age in enumerate(unique_ages)}
    occupation_to_idx = {occupation: idx for idx, occupation in enumerate(unique_occupations)}
    
    # 统计物品特征的所有可能值
    print("统计物品特征...")
    # 提取年份
    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)')[0].astype(int)
    unique_years = sorted(movies_df['year'].unique())
    
    # 提取所有genres
    all_genres = set()
    for genres_str in movies_df['genres']:
        genres = genres_str.split('|')
        all_genres.update(genres)
    all_genres = sorted(list(all_genres))
    
    print(f"年份范围: {min(unique_years)}-{max(unique_years)} (数量: {len(unique_years)})")
    print(f"所有类型: {all_genres} (数量: {len(all_genres)})")
    
    # 创建物品特征映射
    year_to_idx = {year: idx for idx, year in enumerate(unique_years)}
    genre_to_idx = {genre: idx for idx, genre in enumerate(all_genres)}
    
    # 计算特征维度
    user_feature_dim = len(unique_genders) + len(unique_ages) + len(unique_occupations)
    item_feature_dim = len(unique_years) + len(all_genres)
    total_nodes = len(user_id_map) + len(item_id_map) + 1  # +1 for index 0
    feature_dim = user_feature_dim + item_feature_dim
    
    print(f"用户特征维度: {user_feature_dim} (性别:{len(unique_genders)} + 年龄:{len(unique_ages)} + 职业:{len(unique_occupations)})")
    print(f"物品特征维度: {item_feature_dim} (年份:{len(unique_years)} + 类型:{len(all_genres)})")
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
        # Gender特征
        gender_idx = gender_to_idx[user_info['gender']]
        feature_vector[offset + gender_idx] = 1
        offset += len(unique_genders)
        # Age特征
        age_idx = age_to_idx[user_info['age']]
        feature_vector[offset + age_idx] = 1
        offset += len(unique_ages)
        # Occupation特征
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
        
        # Year特征
        year_idx = year_to_idx[movie_info['year']]
        feature_vector[offset + year_idx] = 1
        offset += len(unique_years)
        
        # Genres特征
        genres = movie_info['genres'].split('|')
        for genre in genres:
            if genre in genre_to_idx:
                genre_idx = genre_to_idx[genre]
                feature_vector[offset + genre_idx] = 1
        
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
        edges_features[row.idx, row.rating - 1] = 1  # idx从1开始
    # 保存边特征
    np.save(f"{output_dir}/edges.npy", edges_features)
    print(f"edges.npy已保存，形状: {edges_features.shape}")
    
    print("预处理完成！")
    print(f"输出文件保存在: {output_dir}")
    print("生成的文件:")
    print("- graph.csv: 图结构数据")
    print("- id_mapping.csv: ID映射关系")
    print("- nodes.npy: 节点特征")
    print("- edges.npy: 边特征")

if __name__ == "__main__":
    preprocess_movielens()
