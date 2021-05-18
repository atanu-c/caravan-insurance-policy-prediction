prediction_summary <- function (model, data, cutoff, target_col_name) {
  target_col_index <- grep(target_col_name, colnames(data))
  glm_prob <- predict.glm(model,data[,-target_col_index],type="response")
  glm_predict <- rep(0,nrow(data))
  glm_predict[glm_prob>cutoff] <- 1
  table(pred=glm_predict,true=data[, target_col_index])
  conf_mat <- confusionMatrix(as.factor(glm_predict), as.factor(data[, target_col_index]))
  roc_obj <- roc(data[, target_col_index],glm_predict)
  auc_plot <- plot(roc_obj, print.auc=TRUE)
  #cat("AUC:",auc(roc_obj))
  #auc_value <- auc(roc_obj)
  return(list("auc_value" = auc(roc_obj), "auc_plot" = auc_plot, "conf_mat" = conf_mat))
}