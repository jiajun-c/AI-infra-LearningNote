# 集合通信算法实现

## 1. AllReduce 

## 1.1 Tree

## 1.2 Ring

## 1.3 CollNet

Collective Network算法，Coll指的是节点内的通信操作，Net是用于节点间通信的插件

CollNet 将通信操作分为节点内通信和节点间通信两个部分，在节点内将GPU中数据先AllReduce到Head GPU上，然后将做跨节点的AllReduce，最后在节点内广播和Gather数据
 