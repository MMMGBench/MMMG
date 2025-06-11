import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import copy
import math


def parse_dependencies(dependencies):
    '''
    解析依赖关系：
      - Causes(e_i, e_j): e_i → e_j
      - Requires(e_j, e_k): e_j ← e_k
      - Entails(e_p, e_q): e_p (true) → e_q (true)
      - Defines(e_i, e_j): e_i ↔ e_j
      - TemporalOrder(e_i, e_j): e_i → e_j

      有可能有+ ,分隔
    '''
    edges = []
    for dep, exists in dependencies.items():
        if exists:
            try:
                relation, nodes = dep.split('(', 1)
                try:
                    if nodes[-1] == ")":
                        nodes = nodes[:-1]
                    else:
                        print(f"Error parsing dependency BRACKETS: {dep}")

                    source, target = nodes.split(', ', 1)
                        
                except ValueError:
                    if len(nodes.split(', '), 1) == 1 and len(nodes.split(','), 1) == 2:
                        source, target = nodes.split(',', 1)
                    else:
                        print(f"Error parsing dependency: {dep}")
                        continue
                    
                source = source.lower()
                target = target.lower()

                if "change(" in source.lower():
                    source = source.strip(")").split("change(")[-1].lower()
                if "change(" in target.lower():
                    target = target.strip(")").split("change(")[-1].lower() 
                
                lower_relation = relation.lower()
                if lower_relation == "requires":
                    if isinstance(target, list) or isinstance(target, tuple):
                        for t in target:
                            edges.append((t, source, relation))
                    else:
                        edges.append((target, source, relation))
                elif lower_relation == "defines":
                    if isinstance(target, list) or isinstance(target, tuple):
                        for t in target:
                            edges.append((source, t, relation))
                            edges.append((t, source, relation))
                    else:
                        edges.append((source, target, relation))
                        edges.append((target, source, relation))
                    
                else:
                    if isinstance(target, list) or isinstance(target, tuple):
                        for t in target:
                            edges.append((source, t, relation))
                    else:
                        edges.append((source, target, relation))
            except ValueError:
                continue  # 跳过格式错误的项
    return edges


def build_graph(elements, dependencies):
    G = nx.DiGraph()
    for node, exists in elements.items():
        if exists:
            G.add_node(node.lower())
    edges = parse_dependencies(dependencies)

    
    for source, target, relation in edges:
        match_src = False
        match_tgt = False
        # 当 nodes更长的时候有可能能匹配，有可能是不同的词，要看是否存在" "可以避免大部分
        src_candidate = [s for s in G.nodes if source in s and source != s] + [s for s in G.nodes if s in source and s != source and " " in source]
        tgt_candidate = [t for t in G.nodes if target in t and target != t] + [t for t in G.nodes if t in target and t != target and " " in target]
        
        #避免source/target是candidate中的词元，本身candidate不是此意，如果不去除可能导致错误匹配
        src_candidate = [s for s in src_candidate if "" in s]
        tgt_candidate = [t for t in tgt_candidate if "" in t]
        #首先严格匹配
        for node in G.nodes:
            if source == node:
                match_src = True
            if target == node:
                match_tgt = True
        
        if not match_src:
            # 如果没有严格匹配，尝试模糊匹配
            if len(src_candidate) == 1:
                match_src = True
                source = src_candidate[0]
            # elif len(src_candidate) > 1:
            #     print(f"Warning: Multiple candidates for source '{source}': {src_candidate}.")
        if not match_tgt:
            if len(tgt_candidate) == 1:
                match_tgt = True
                target = tgt_candidate[0]
            # elif len(tgt_candidate) > 1:
            #     print(f"Warning: Multiple candidates for target '{target}': {tgt_candidate}.")

        # 如果source和target都匹配，则添加边
        if match_src and match_tgt:
            G.add_edge(source, target, label=relation)
          
    return G

def normalized_ged(G1, G2):
    """归一化图编辑距离"""
    try:
        ged = next(nx.optimize_graph_edit_distance(G1, G2))
    except StopIteration:
        ged = 0  # 若没有迭代器内容
    max_size = G1.number_of_nodes() + G2.number_of_nodes() + G1.number_of_edges() + G2.number_of_edges()
    return ged / max_size if max_size > 0 else 1.0

def mcs_similarity(G1, G2):
    """最大公共子图相似度"""
    matcher = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
    if matcher.subgraph_is_isomorphic():
        mcs_size = len(matcher.mapping)
        max_size = max(G1.number_of_nodes(), G2.number_of_nodes())
        return mcs_size / max_size if max_size > 0 else 1.0
    else:
        return 0.0

def jaccard_similarity(G1, G2):
    """Jaccard 相似度（节点集合）"""
    nodes1 = set(G1.nodes())
    nodes2 = set(G2.nodes())
    intersection = nodes1 & nodes2
    union = nodes1 | nodes2
    return len(intersection) / len(union) if union else 1.0

def jaccard_edge_similarity(G1, G2):
    """Jaccard 相似度（边集合）"""
    edges1 = set(frozenset(edge) for edge in G1.edges())
    edges2 = set(frozenset(edge) for edge in G2.edges())
    intersection = edges1 & edges2
    union = edges1 | edges2
    return len(intersection) / len(union) if union else 1.0

def calc_weighting(s):
    if s <= 70:
        return 1
    else:
        ss = max((160 - s) / 90, 0)
        return ss

def graph_to_vector(G, all_nodes):
    """图转成二值向量"""
    return np.array([1 if node in G.nodes else 0 for node in all_nodes])

def process_data(score_dir, data_dir, visualize_dir):
    visualize = False
    if visualize_dir is not None:
        visualize=True
    with open(score_dir, "r", encoding="utf-8") as f:
        score_data = json.load(f)
    with open(data_dir, "r", encoding="utf-8") as f:
        data = json.load(f)
    total_statistics = {"Details": {}, "SchoolAVG": {}, "DisciplineAVG": {}}
    for school in data.keys():
        classes_metrics = {}

        level_data = data[school]
        for element in level_data.keys():
            discipline = element.split("_")[1]
            try:
                k_score_meta = score_data[element]["region_count"]
            except KeyError:
                print(f"KeyError: {element} not found in score_data.")
                k_score_meta = 0
                
            if discipline not in classes_metrics:
                classes_metrics[discipline] = {"cnt":0, "GED": 0, "Jaccard": 0, "Jaccard Edge": 0, "K-score(w)": 0}
            branch_data = level_data[element]["result"]
            all_elements = {n: True for n in branch_data["elements"].keys()}
            all_dependencies = {dep: True for dep in branch_data["dependencies"].keys()}
            G_gt = build_graph(all_elements, all_dependencies)
            G_data = build_graph(branch_data["elements"], branch_data["dependencies"])

            ged_gt_data   = normalized_ged(G_gt, G_data)
            jaccard_gt_data   = jaccard_similarity(G_gt, G_data)
            jaccard_edge_gt_data   = jaccard_edge_similarity(G_gt, G_data)
            R_score = calc_weighting(k_score_meta)
            k_score_w = R_score * (1-ged_gt_data)

            classes_metrics[discipline]["GED"] += ged_gt_data
            classes_metrics[discipline]["Jaccard"] += jaccard_gt_data
            classes_metrics[discipline]["Jaccard Edge"] += jaccard_edge_gt_data
            classes_metrics[discipline]["cnt"] += 1
            classes_metrics[discipline]["K-score(w)"] += k_score_w
            
            #pos_gt = nx.spring_layout(G_gt, seed=42)
            if visualize:
            # GT 与 4o 图
                fig, axes = plt.subplots( figsize=(10, 10)) # add subplot for flux_recap
                optimal_k = 5.5 / math.sqrt(len(G_gt.nodes)+1)
                pos_gt = nx.spring_layout(G_gt, k=optimal_k, iterations=200, seed=42)
            
                nx.draw(G_gt, pos_gt, ax=axes, node_color='lightgrey', edge_color='grey',
                        node_size=1500, font_size=10, alpha=0.5, with_labels=True)
                nx.draw(G_data, pos_gt, ax=axes, node_color='lightblue', edge_color='blue',
                        node_size=1500, font_size=10, with_labels=True)
                axes.set_title(f'GT vs data\n'
                                f'GED: {ged_gt_data:.2f}, R-score: {R_score:.2f}, K-score: {k_score_w:.2f}') #Jaccard: {jaccard_gt_data:.2f}, 'f'Jaccard Edge: {jaccard_edge_gt_data:.2f},

                plt.tight_layout()
    
                # 保存图片到目标文件夹
                if not os.path.exists(visualize_dir):
                    os.makedirs(visualize_dir)
                img_path = os.path.join(visualize_dir, f"{school}__{element}__graph_compare.png")
                plt.savefig(img_path)
                plt.close(fig)
        record_classes_metrics = copy.deepcopy(classes_metrics)
        total_statistics["Details"][school] = record_classes_metrics
        # 计算当前学段平均值
        school_avg = {"GED": 0, "Jaccard": 0, "Jaccard Edge": 0, "K-score(w)": 0, "cnt": 0}
        for discipline in classes_metrics.keys():
            school_avg["GED"] += classes_metrics[discipline]["GED"]
            school_avg["Jaccard"] += classes_metrics[discipline]["Jaccard"]
            school_avg["Jaccard Edge"] += classes_metrics[discipline]["Jaccard Edge"]
            school_avg["K-score(w)"] += classes_metrics[discipline]["K-score(w)"]
            school_avg["cnt"] += classes_metrics[discipline]["cnt"]
        if school_avg["cnt"] > 0:
            school_avg["GED"] /= school_avg["cnt"]
            school_avg["Jaccard"] /= school_avg["cnt"]
            school_avg["Jaccard Edge"] /= school_avg["cnt"]
            school_avg["K-score(w)"] /= school_avg["cnt"]

        total_statistics["SchoolAVG"][school] = school_avg

    # 计算所有学科平均值
    for items in total_statistics["Details"].values():
        for discipline, metrics in items.items():
            if discipline not in total_statistics["DisciplineAVG"]:
                total_statistics["DisciplineAVG"][discipline] = {"cnt":0, "GED": 0, "Jaccard": 0, "Jaccard Edge": 0, "K-score(w)": 0}
            total_statistics["DisciplineAVG"][discipline]["K-score(w)"] += metrics["K-score(w)"]
            total_statistics["DisciplineAVG"][discipline]["GED"] += metrics["GED"]
            total_statistics["DisciplineAVG"][discipline]["Jaccard"] += metrics["Jaccard"]
            total_statistics["DisciplineAVG"][discipline]["Jaccard Edge"] += metrics["Jaccard Edge"]
            total_statistics["DisciplineAVG"][discipline]["cnt"] += metrics["cnt"]

    # 计算平均值
    for discipline, metrics in total_statistics["DisciplineAVG"].items():
        cnt = metrics["cnt"]
        if cnt > 0:
            total_statistics["DisciplineAVG"][discipline]["GED"] /= cnt
            total_statistics["DisciplineAVG"][discipline]["Jaccard"] /= cnt
            total_statistics["DisciplineAVG"][discipline]["Jaccard Edge"] /= cnt
            total_statistics["DisciplineAVG"][discipline]["K-score(w)"] /= cnt
        else:
            total_statistics["DisciplineAVG"][discipline]["GED"] = 0
            total_statistics["DisciplineAVG"][discipline]["Jaccard"] = 0
            total_statistics["DisciplineAVG"][discipline]["Jaccard Edge"] = 0
            total_statistics["DisciplineAVG"][discipline]["K-score(w)"] = 0

    data["statistics"] = total_statistics
    # 将统计结果写入 JSON 文件
    with open(data_dir.split(".json")[0]+"_recaptioned.json", "w", encoding="utf-8") as f:
        json.dump(total_statistics, f, indent=4, ensure_ascii=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON data and visualize graphs.")
    parser.add_argument("--score_dir", type=str, required=True, help="Path t9o the Stage1 JSON file: knowledge fidelity")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the Stage2 JSON file: visual readability")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to the MMMG Statistics JSON file.")
    parser.add_argument("--visualize_dir", type=str, default=None, help="Path to the destination folder for images.")
   
    args = parser.parse_args()
    
    process_data(args.score_dir, args.data_dir, args.visualize_dir)