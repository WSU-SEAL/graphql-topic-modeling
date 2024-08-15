library("doBy")

library(dplyr)
library(nortest)


#READ TOSEM_DATASET.csv
chartds = read.csv(file.choose(), header = TRUE)

acceptedDF <- chartds %>%
  filter(has_accepted_answer ==1)

acceptedDF$answer_interval <- as.numeric(acceptedDF$accepted_answer_time)


#Normality test

ad.test(chartds$ViewCount)
ad.test(acceptedDF$answer_interval)
ad.test(chartds$Score)
ad.test(chartds$discussion_count)



# Group by statistics
summaryBy(ViewCount ~ name, data = chartds, 
          FUN = list(median))

summaryBy(has_accepted_answer ~ name, data = chartds, 
          FUN = list(mean,  sum, length))


summaryBy(is_answered ~ name, data = chartds, 
          FUN = list(mean,  sum, length))



summaryBy(answer_interval ~ name, data = acceptedDF,
          FUN = list(median))



summaryBy(Score ~ name, data = chartds, 
          FUN = list(mean,median))

summaryBy(discussion_count ~ name, data = chartds, 
          FUN = list(mean))


summaryBy(AnswerCount ~ name, data = chartds, 
          FUN = list(mean))


#Overall 
median(chartds$ViewCount)
mean(chartds$Score)
mean(chartds$has_accepted_answer)
mean(chartds$is_answered)
median(acceptedDF$answer_interval)
chartds$no_accepted_answer<- with(chartds,1-has_accepted_answer)
chartds$no_upvoted_answer<- with(chartds,1-is_answered)

#difficulty popularity correlation
cor.test(chartds$ViewCount, chartds$no_accepted_answer, method = c("pearson"))
cor.test(chartds$Score, chartds$no_accepted_answer, method = c("kendall"))
cor.test(chartds$discussion_count, chartds$no_accepted_answer, method = c("kendall"))

cor.test(chartds$ViewCount, chartds$no_upvoted_answer, method = c("kendall"))
cor.test(chartds$Score, chartds$no_upvoted_answer, method = c("kendall"))
cor.test(chartds$discussion_count, chartds$no_upvoted_answer, method = c("kendall"))

cor.test( acceptedDF$answer_interval, acceptedDF$ViewCount, method = c("kendall"))
cor.test( acceptedDF$answer_interval, acceptedDF$Score, method = c("kendall"))
cor.test( acceptedDF$answer_interval, acceptedDF$discussion_count, method = c("kendall"))


summaryBy(ViewCount ~ category, data = chartds, 
          FUN = list(median))

summaryBy(has_accepted_answer ~ category, data = chartds, 
          FUN = list(mean,  sum, length))


summaryBy(is_answered ~ category, data = chartds, 
          FUN = list(mean,  sum, length))



summaryBy(answer_interval ~ category, data = acceptedDF,
          FUN = list(median))



summaryBy(Score ~ category, data = chartds, 
          FUN = list(mean))

summaryBy(discussion_count ~ category, data = chartds, 
          FUN = list(mean))






