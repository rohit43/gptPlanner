library(shiny)
library(ggplot2)
library(dplyr)
library(scales)
library(DT)

calculateGpt <- function(numHiddenLayers, dModel, numHeads, dFf, seqLength, batchSize, vocabSize, dtypeSize = 4, optimizer = "Adam", activation = "ReLU") {
  vocabSize <- 50257
  tokenEmbeddingParams <- dModel * vocabSize
  positionalEmbeddingParams <- dModel * seqLength
  attnParamsPerLayer <- 4 * dModel * dModel
  ffnParamsPerLayer <- 2 * dModel * dFf
  layerNormParamsPerLayer <- 4 * dModel
  paramsPerHiddenLayer <- attnParamsPerLayer + ffnParamsPerLayer + layerNormParamsPerLayer
  totalHiddenLayersParams <- numHiddenLayers * paramsPerHiddenLayer
  finalLayerNormParams <- 2 * dModel
  outputLayerParams <- dModel * vocabSize
  totalParams <- tokenEmbeddingParams + positionalEmbeddingParams +
    totalHiddenLayersParams + finalLayerNormParams + outputLayerParams
  paramMemory <- totalParams * dtypeSize
  activationMemory <- batchSize * seqLength * dModel * (numHiddenLayers + 1) * dtypeSize
  gradientMemory <- paramMemory
  optimizerMemory <- if (optimizer == "Adam") paramMemory * 2 else paramMemory
  totalMemory <- paramMemory + activationMemory + gradientMemory + optimizerMemory
  attnFlops <- 4 * numHeads * seqLength^2 * dModel
  ffnFlops <- 2 * seqLength * (dModel * dFf)
  totalFlops <- numHiddenLayers * (attnFlops + ffnFlops)
  modelSizeGB <- paramMemory / (1024^3)
  paramBreakdown <- data.frame(
    Component = c("Token Embeddings", "Positional Embeddings",
                  paste("Hidden Layers (", numHiddenLayers, ")", sep=""),
                  "Final Layer Norm", "Output Layer"),
    Parameters = c(tokenEmbeddingParams, positionalEmbeddingParams,
                   totalHiddenLayersParams, finalLayerNormParams, outputLayerParams)
  )
  memoryBreakdown <- data.frame(
    Component = c("Parameters", "Activations", "Gradients", "Optimizer"),
    Size = c(paramMemory, activationMemory, gradientMemory, optimizerMemory) / (1024^3)  # Convert to GB
  )
  hiddenLayerBreakdown <- data.frame(
    Component = c("Self-Attention", "Feed-Forward Network", "Layer Normalization"),
    ParametersPerLayer = c(attnParamsPerLayer, ffnParamsPerLayer, layerNormParamsPerLayer),
    TotalParameters = c(attnParamsPerLayer, ffnParamsPerLayer, layerNormParamsPerLayer) * numHiddenLayers
  )
  details <- list(
    dModel = dModel,
    numHeads = numHeads,
    dFf = dFf,
    seqLength = seqLength,
    vocabSize = vocabSize,
    numHiddenLayers = numHiddenLayers,
    paramsPerHead = dModel / numHeads,
    totalAttentionHeads = numHeads * numHiddenLayers,
    optimizer = optimizer,
    activation = activation
  )
  return(list(
    totalParams = totalParams,
    paramBreakdown = paramBreakdown,
    totalMemory = totalMemory / (1024^3),
    memoryBreakdown = memoryBreakdown,
    hiddenLayerBreakdown = hiddenLayerBreakdown,
    details = details,
    flops = totalFlops,
    modelSizeGB = modelSizeGB
  ))
}
ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
      body {
        font-family: 'Roboto', sans-serif;
        background-color: #f0f5f9;
        color: #333;
      }
      .container-fluid {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
      }
      h2 {
        color: #1e3d59;
        border-bottom: 2px solid #1e3d59;
        padding-bottom: 10px;
      }
      h3 {
        color: #1e3d59;
        font-size: 12px;  /* Reduced font size */
        border-bottom: 2px solid #1e3d59;
        padding-bottom: 10px;
      }
      .well {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
      }
      .form-control {
        border-radius: 5px;
        background-color: #e3e8ee;
      }
      .btn-primary {
        background-color: #ff6e40;
        border-color: #ff6e40;
        transition: all 0.3s ease;
        color: white;
        font-weight: bold;
      }
      .btn-primary:hover {
        background-color: #ff5722;
        border-color: #ff5722;
      }
      #totalParams, #totalMemory, #modelSize, #flops {
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        color: #1e3d59;
      }
      .plot-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      }
      .plot-container h3 {
        color: #1e3d59;
      }
      #footer {
        text-align: center;
        margin-top: 20px;
        padding: 10px;
        font-size: 12px;
        color: #666;
      }
      .explanation {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 15px;
        margin-top: 20px;
        border-radius: 5px;
        font-size: 16px;
        line-height: 1.6;
        color: #333;
      }
    "))
  ),
  titlePanel("GPT Planner"),
  h3("Still in the making. Please use with caution. "),
  sidebarLayout(
    sidebarPanel(
      numericInput("numHiddenLayers", "Number of Hidden Layers", value = 12, min = 1, max = 100),
      numericInput("dModel", "Model Dimension", value = 768, min = 64, max = 4096),
      numericInput("numHeads", "Attention Heads", value = 12, min = 1, max = 64),
      numericInput("dFf", "Feed-forward Dim", value = 3072, min = 256, max = 16384),
      numericInput("seqLength", "Sequence Length", value = 512, min = 1, max = 4096),
      numericInput("batchSize", "Batch Size", value = 32, min = 1, max = 1024),
      numericInput("vocabSize", "Vocabulary Size", value = 50257, min = 1000, max = 100000),
      numericInput("dtypeSize", "Data Type (bytes)", value = 4, min = 1, max = 8),
      selectInput("optimizer", "Optimizer", choices = c("Adam", "SGD", "AdamW")),
      selectInput("activation", "Activation Function", choices = c("ReLU", "GELU", "Tanh")),
      actionButton("calculate", "Calculate", class = "btn-primary btn-block")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Overview",
                 fluidRow(
                   column(6, htmlOutput("totalParams")),
                   column(6, htmlOutput("totalMemory")),
                   column(6, htmlOutput("modelSize")),
                   column(6, htmlOutput("flops"))
                 ),
                 fluidRow(
                   column(6, plotOutput("paramBreakdownPlot", height = "350px")),
                   column(6, plotOutput("memoryBreakdownPlot", height = "350px"))
                 ),
                 fluidRow(
                   column(12, htmlOutput("explanationText", class = "explanation"))
                 )
        ),
        tabPanel("Hidden Layer Details",
                 h3("Hidden Layer Breakdown"),
                 plotOutput("hiddenLayerBreakdownPlot", height = "400px"),
                 h3("Model Architecture Details"),
                 tableOutput("modelDetails"),
                 h3("Hidden Layer Parameters"),
                 DTOutput("hiddenLayerParamsTable")
        )
      )
    )
  ),
  tags$div(id = "footer",
           "",
           tags$a(href = "https://rohit43.github.io/", "Rohit Vashisht", target = "_blank")
  )
)

server <- function(input, output) {
  modelData <- reactiveVal(NULL)
  observeEvent(input$calculate, {
    result <- calculateGpt(
      numHiddenLayers = input$numHiddenLayers,
      dModel = input$dModel,
      numHeads = input$numHeads,
      dFf = input$dFf,
      seqLength = input$seqLength,
      batchSize = input$batchSize,
      vocabSize = input$vocabSize,
      dtypeSize = input$dtypeSize,
      optimizer = input$optimizer,
      activation = input$activation
    )
    modelData(result)
  })
  output$totalParams <- renderUI({
    result <- modelData()
    if (!is.null(result)) {
      paste("Total Parameters: ", format(result$totalParams, big.mark = ",", scientific = FALSE))
    }
  })
  output$totalMemory <- renderUI({
    result <- modelData()
    if (!is.null(result)) {
      paste("Total Memory Usage: ", format(round(result$totalMemory, 2), nsmall = 2), " GB")
    }
  })
  output$modelSize <- renderUI({
    result <- modelData()
    if (!is.null(result)) {
      paste("Model Size: ", format(round(result$modelSizeGB, 2), nsmall = 2), " GB")
    }
  })
  output$flops <- renderUI({
    result <- modelData()
    if (!is.null(result)) {
      paste("Total FLOPs: ", format(result$flops, big.mark = ",", scientific = FALSE))
    }
  })
  output$paramBreakdownPlot <- renderPlot({
    result <- modelData()
    if (!is.null(result)) {
      ggplot(result$paramBreakdown, aes(x = reorder(Component, -Parameters), y = Parameters, fill = Component)) +
        geom_bar(stat = "identity", alpha = 0.8) +
        theme_minimal(base_family = "Roboto") +
        scale_fill_manual(values = c("#ff6e40", "#ffccbc", "#ffab91", "#8d6e63", "#4e342e")) +
        scale_y_continuous(labels = comma) +
        labs(title = "Parameter Breakdown by Component", y = "Parameters", x = "") +
        theme(axis.text.x = element_text(angle = 45, hjust = 1),
              legend.position = "none")
    }
  })
  output$memoryBreakdownPlot <- renderPlot({
    result <- modelData()
    if (!is.null(result)) {
      ggplot(result$memoryBreakdown, aes(x = reorder(Component, -Size), y = Size, fill = Component)) +
        geom_bar(stat = "identity", alpha = 0.8) +
        theme_minimal(base_family = "Roboto") +
        scale_fill_manual(values = c("#2196f3", "#64b5f6", "#bbdefb", "#1e88e5")) +
        scale_y_continuous(labels = comma) +
        labs(title = "Memory Breakdown by Component (GB)", y = "Memory (GB)", x = "") +
        theme(axis.text.x = element_text(angle = 45, hjust = 1),
              legend.position = "none")
    }
  })
  output$hiddenLayerBreakdownPlot <- renderPlot({
    result <- modelData()
    if (!is.null(result)) {
      ggplot(result$hiddenLayerBreakdown, aes(x = reorder(Component, -TotalParameters), y = TotalParameters, fill = Component)) +
        geom_bar(stat = "identity", alpha = 0.8) +
        theme_minimal(base_family = "Roboto") +
        scale_fill_manual(values = c("#4caf50", "#81c784", "#a5d6a7")) +
        scale_y_continuous(labels = comma) +
        labs(title = "Hidden Layer Breakdown", y = "Total Parameters", x = "") +
        theme(axis.text.x = element_text(angle = 45, hjust = 1),
              legend.position = "none")
    }
  })
  output$modelDetails <- renderTable({
    result <- modelData()
    if (!is.null(result)) {
      details <- result$details
      detailsTable <- data.frame(
        Parameter = c("Model Dimension", "Attention Heads", "Feed-forward Dim", "Sequence Length", "Vocabulary Size", "Number of Hidden Layers", "Parameters per Head", "Total Attention Heads", "Optimizer", "Activation Function"),
        Value = c(details$dModel, details$numHeads, details$dFf, details$seqLength, details$vocabSize, details$numHiddenLayers, details$paramsPerHead, details$totalAttentionHeads, details$optimizer, details$activation)
      )
      detailsTable
    }
  })
  output$hiddenLayerParamsTable <- renderDT({
    result <- modelData()
    if (!is.null(result)) {
      datatable(result$hiddenLayerBreakdown, options = list(dom = 't'))
    }
  })
  output$explanationText <- renderUI({
    result <- modelData()
    if (!is.null(result)) {
      HTML(paste0(
        "<h4>Model Architecture Overview</h4>",
        "<p>This GPT model features:</p>",
        "<ul>",
        "<li>", result$details$numHiddenLayers, " hidden layers</li>",
        "<li>Model dimension of ", result$details$dModel, "</li>",
        "<li>", result$details$numHeads, " attention heads</li>",
        "<li>Feed-forward dimension of ", result$details$dFf, "</li>",
        "<li>Sequence length of ", result$details$seqLength, "</li>",
        "<li>Vocabulary size of ", result$details$vocabSize, "</li>",
        "</ul>",

        "<h4>Model Scale</h4>",
        "<p>The model contains ", format(result$totalParams, big.mark = ",", scientific = FALSE), " parameters in total. ",
        "The majority of these are distributed across the hidden layers, with significant contributions from token embeddings and the output layer.</p>",

        "<h4>Memory Requirements</h4>",
        "<p>During training, this model will requires approximately ", format(round(result$totalMemory, 2), nsmall = 2), " GB of memory. ",
        "This includes space for parameters, activations, gradients, and optimizer states. ",
        "The model size on disk is about ", format(round(result$modelSizeGB, 2), nsmall = 2), " GB.</p>",

        "<h4>Computational Demand</h4>",
        "<p>The model requires ", format(result$flops, big.mark = ",", scientific = FALSE), " floating-point operations (FLOPs) per forward pass. </p>",

        "<h4>Training Configuration</h4>",
        "<p>The model uses the ", result$details$optimizer, " optimizer and ", result$details$activation, " activation function, ",
        "which influence its training dynamics and performance.</p>"
      ))
    }
  })
}
shinyApp(ui = ui, server = server)
