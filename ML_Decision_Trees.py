'''

PA-1: Decision Trees
Authors:
Amitabh Rajkumar Saini, amitabhr@usc.edu
Shilpa Jain, shilpaj@usc.edu
Sushumna Khandelwal, sushumna@usc.edu

Dependencies: 
1. numpy : pip install numpy
2. graphviz : pip install graphviz
3. graphviz for OS : https://graphviz.gitlab.io/download/

Output:
Returns a decision tree graph as PNG File and the prediction on console.

'''

import numpy as np
import re
from graphviz import Digraph
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

class decision_node:
    '''
    Decision Node defines a single node of the tree containing all meta-data
    '''
    def __init__(self,df,parent):   
        '''
        Constructs a decision_node
        :param df: Dataframe of type numpy 2D Array
        :param parent: Parent of this node of type decision_node
        :return: returns nothing
        '''
        self.df = df[1:]
        self.attributes = df[0]
        self.df_entropy = float('-inf')
        self.infogain = float('-inf')
        self.parent_node = parent
        self.children_nodes = dict()
        self.node_name = ""
  

    def entropy(self,label_index,attribute_index=None):
        '''
        Calculates entropy of a particular attribute or the whole dataframe
        :param label_index: index of label column 
        :param attribute_index: index of attribute column 
        :return: returns entropy
        '''
        if attribute_index == None:
            values,count = np.unique(self.df[:,label_index],return_counts = True)
            e = 0
            for each in count:
                p = each/np.sum(count)
                e -= p * np.log2(p)
            return e
        else:
            values,count = np.unique(self.df[:,attribute_index], return_counts=True)
            e_attr = 0
            attr_value_indices={}
            for i in range(count.size):
                p_attr = count[i]/np.sum(count)
                indices = np.where(self.df[:,attribute_index]==values[i])
                attr_value_indices[values[i]]=indices
                attr_label,attr_label_count = np.unique(np.take(self.df[:,label_index],indices),return_counts=True)
                e = 0
                for each in attr_label_count:
                    p = each/np.sum(attr_label_count)
                    e -= p * np.log2(p)
                e_attr += p_attr*e
            return e_attr, attr_value_indices
       
    def calcinfogain(self,cur_df_entropy,attr_entropy):
        '''
        Calculates information gain
        :param cur_df_entropy: Entropy of DataFrame 
        :param attr_entropy: Entropy of the Attribute
        :return: returns information gain
        '''
        return cur_df_entropy-attr_entropy

    def get_best_split(self):
        '''
        Finds the best attribute to split the data using information gain
        :return: returns dataframe after splitting on the attribute and the attribute
        '''
        max_gain = float('-inf')
        max_attribute_index = -1
        max_attr_df_indices = np.asarray([])
        for i in range(self.df.shape[1]-1):           #no.of cols
            attr_entropy, attr_df_indices = self.entropy(-1,i)
            self.df_entropy = self.entropy(-1)
            gain = self.calcinfogain(self.df_entropy,attr_entropy)
            if max_gain < gain:
                max_gain = gain
                max_attribute_index = i
                max_attr_df_indices = attr_df_indices 
        self.infogain = max_gain
        if self.infogain == 0 or self.infogain == float('-inf'):
            values,count = np.unique(self.df[:,-1], return_counts=True)
            maxcount = float('-inf')
            for i in range(count.size):
                if maxcount<count[i]:
                    maxcount=count[i]
                    self.node_name = values[i]

            return None,[]
        self.node_name = self.attributes[max_attribute_index]
        return max_attribute_index, max_attr_df_indices

    def create_child(self,col_index,attr_name,attr_df_indices):
        '''
        Creates a child node
        :param col_index: index of the attribute column used for splitting
        :param attr_name: attribute value for which the node will be created
        :param attr_df_indices: indices of the rows containig the attr_name
        :return: returns child node
        '''
        t_df = self.df[attr_df_indices]
        t_df = np.delete(t_df,col_index,axis=1)
        h_df = np.delete(self.attributes,col_index)
        x = []
        x.append(h_df)
        df = np.append(np.asarray(x),t_df,axis=0)
        child_node = decision_node(df,self)
        self.children_nodes[attr_name] = child_node
        return child_node

class decision_tree:
    '''
    Class contains methods to create the tree using ID3 Algorithm, predict and export the tree as directed graph
    '''
    def __init__(self):
        '''
        Creates a decision tree instance
        :return: returns nothing
        '''
        self.full_df = None
        self.root = None
        #self.attributes = None

    def txt_to_df(self,filepath):
        '''
        Opens txt file and makes numpy 2d array
        :return: returns numpy 2d array
        '''
        f=open(filepath,"r")
        lines = f.readlines()

        def clean_data(each_line):
            each_line = each_line.split(":")[-1]
            each_line = re.sub('[^0-9a-zA-Z,\- ]+', '', each_line)
            return each_line
        cleaned_data = []
        for i, each in enumerate(lines):
            if not re.match(r'^\s*$', each):
                cleaned_data.append(clean_data(each))
        df = []
        for i in range(len(cleaned_data)):
            df.append(cleaned_data[i].strip().replace(", ",",").split(","))
        #print(df)
        return np.asarray(df)

    def csv_to_df(self,filepath):
        '''
        Opens csv file and makes numpy 2d array
        :return: returns numpy 2d array
        '''
        f=open(filepath,"r")
        lines = f.readlines()

        df = []
        for i in range(len(lines)):
            df.append(lines[i].strip().split(","))
        return np.asarray(df)
    
    def create_tree(self):
        '''
        Creates decision tree using ID3 algorithm iteratively
        :return: returns nothing
        '''
        self.root = decision_node(self.full_df,None)
        q = []
        q.append(self.root)
        while(q):
            d_node = q.pop(0)
            attr_index, max_attr_df_indices = d_node.get_best_split()
            for each in max_attr_df_indices:
                q.append(d_node.create_child(attr_index,each,max_attr_df_indices[each]))
            #print(d_node.parent_node,d_node.node_name,d_node.children_nodes.keys())
            #print("\n")
    
    def print_tree(self):
        '''
        Prints decision tree on console
        :return: returns nothing
        '''
        q = []
        q.append((self.root,None))
        while(q):
            x = q.pop(0)
            p = x[0].parent_node.node_name if x[0].parent_node!=None else 'None'
            print(p,x[0].node_name,x[0].infogain,x[1])
            for each in x[0].children_nodes:
                q.append((x[0].children_nodes[each],each))
          
    def predict(self,test_df):
        '''
        Predicts using the decision tree generated
        :param test_df: numpy 2d array to be predicted
        :return: returns nothing
        '''
        header = test_df[0]
        out_answers = np.unique(self.root.df[:,-1])
        pred_answers = []
        for row in range(1,len(test_df)):
            current = self.root
            row_data = test_df[row]
            while current.node_name not in out_answers:
                index = np.where(header==current.node_name)
                current = current.children_nodes[row_data[index[0][0]]]
            pred_answers.append(current.node_name)
        return pred_answers

    def creategraph(self):
        '''
        Creates decision tree using graphviz for visually
        :return: returns nothing
        '''
        graph = Digraph()
        q=[]
        q.append(self.root)
        graph.node(str(self.root),self.root.node_name+"\nIG="+str(round(self.root.infogain,2))+"\nEntropy="+str(round(self.root.df_entropy,2)))
        while q:
            x = q.pop(0)
            for each in x.children_nodes:
                graph.node(str(x.children_nodes[each]),x.children_nodes[each].node_name+"\nIG="+str(round(x.children_nodes[each].infogain,2))+"\nEntropy="+str(round(x.children_nodes[each].df_entropy,2)))
                graph.edge(str(x),str(x.children_nodes[each]),label=each)
                q.append(x.children_nodes[each])

        graph.render('DecisionTree.gv',view=True,format='png')


def metrics(actual,predicted):
    '''
    Calculates metrics
    :param actual: actual values in list
    :param predicted: model predicted values in list
    :return : returns nothing
    '''
    confusion_matrix=[[0,0],[0,0]]
    for i in range(len(actual)):
        actual[i]=1 if actual[i]=="Yes" else 0
        predicted[i]=1 if predicted[i]=="Yes" else 0

        if actual[i] and predicted[i]:
            confusion_matrix[0][0]+=1
        elif actual[i] and not predicted[i]:
            confusion_matrix[0][1]+=1
        elif not actual[i] and predicted[i]:
            confusion_matrix[1][0]+=1
        else:
            confusion_matrix[1][1]+=1

    accuracy=(confusion_matrix[1][1]+confusion_matrix[0][0])/(sum(confusion_matrix[0])+sum(confusion_matrix[1]))
    recall=(confusion_matrix[0][0])/sum(confusion_matrix[0])
    precision=(confusion_matrix[0][0])/(confusion_matrix[0][0]+confusion_matrix[1][0])
    Fmeasure=(2*recall*precision)/(recall+precision)
    print("Accuracy: ",accuracy)
    print("Recall: ",recall)
    print("Precision: ",precision)
    print("Fmeasure: ",Fmeasure)
    print("=======================")
    Matrix=[]
    Matrix.append(["","P1","N0"])
    Matrix.append((["P1"]+confusion_matrix[0]))
    Matrix.append((["N0"]+confusion_matrix[1]))
    for i in Matrix:
        print(i)
    print("TP: ",confusion_matrix[0][0])
    print("FP: ",confusion_matrix[0][1])
    print("FN: ",confusion_matrix[1][0])
    print("TN: ",confusion_matrix[1][1])

def main():
    '''
    Runner Program
    :return: returns nothing
    '''
    d = decision_tree()
    d.full_df = d.txt_to_df("train.txt")
    d.create_tree()
    #d.print_tree()
    #print("------------------------")
    pred_df = d.csv_to_df("predict.csv")
    pred = d.predict(pred_df)
    print(pred)
    if pred_df[0,-1] == d.full_df[0,-1]:
        metrics(list(pred_df[1:,-1]),pred)
    d.creategraph()
 
if __name__== "__main__":
    main()
    