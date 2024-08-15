library("doBy")

library("ggplot2")
library("nortest")
library("scale")
datafile="./posts.csv"


#print(datafile)
#DS = read.csv(file.choose(), header = TRUE)

chartds = read.csv(file.choose(), header = TRUE)

chartds$catFactor=factor(chartds$category)

chartds$year_quarterF=factor(chartds$year_quarter)



#summaryBy(ViewCount ~ name, data = DS, 
#          FUN = list(mean, max, min, median, sd))

ad.test(DS$ViewCount)

ggplot(data = chartds, mapping=aes(x=year_quarterF, y=topic_posts, group=catFactor, color=catFactor)) +
  geom_line() +
  scale_x_discrete(breaks = levels(chartds$year_quarterF)[c(T, rep(F, 3))])+
  xlab("") +
  ylab("Absolute impact") +
  theme(axis.text.x = element_text(angle = 75, vjust =0.5, hjust=.45), 
        legend.title=element_blank(),
        legend.direction = "horizontal", legend.position = "top",
        legend.text = element_text(size = 10,face="bold"))

chartds$relative =chartds$topic_posts/chartds$total_posts

 
ggplot(data = chartds, mapping=aes(x=year_quarterF, y=relative, group=catFactor, color=catFactor)) +
  geom_line() +
  xlab("") +
  ylab("Relative impact") + scale_y_continuous(labels = scales::percent)+
  scale_x_discrete(breaks = levels(chartds$year_quarterF)[c(T, rep(F, 3))])+
  theme(axis.text.x = element_text(angle = 75, vjust =0.5, hjust=.45), 
        legend.title=element_blank(),
        legend.direction = "horizontal", legend.position = "top",
        legend.text = element_text(size = 10,face="bold"))
  
#geom_label(aes(label = topicNameLabel), nudge_x = 0.5, nudge_y= yearlyTrend$topicLabelYNudge, size = 1.5) +


topicds = read.csv(file.choose(), header = TRUE)


ggplot(data = topicds, mapping=aes(x=topiccount, y=cv)) +
  geom_line() +
  xlab("# of topics") +  scale_x_continuous(n.breaks = 9)  + ylim(0.45, 0.6)+
  ylab("CV score")  +
  geom_vline(xintercept = max(14, na.rm=TRUE),
                                color="red") 

ggplot(data = topicds, mapping=aes(x=topiccount, y=stability)) +
  geom_line() +
  xlab("# of topics") +  scale_y_continuous(labels = scales::percent)+
  ylab("LDA stability")  
