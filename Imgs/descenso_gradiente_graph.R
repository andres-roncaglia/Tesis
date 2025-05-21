#-----------------------------------------------

library(plotly)

# Dominio
x <- seq(-7, 7, length.out = 100)
y <- seq(-7, 7, length.out = 100)
alpha <- 0.1  # Suavidad

# Función
z_func <- function(x, y) {
  0.1 * exp(-alpha * ((x - 1)^2 + (y - 1)^2)) -
    0.1 * exp(-alpha * ((x + 1)^2 + (y + 1)^2))
}

z <- outer(x, y, Vectorize(z_func))

# Escala de colores
colorscale_custom <- list(
  c(0, "#2F6F9D"),       # para el valor mínimo
  c(0.2, "#85DCFF"),       # para el valor mínimo
  c(0.5, "#B5E3C4"),    # para valor medio
  c(0.7, "#FBC489"),    # para valor medio
  c(1, "#D5202C")         # para el valor máximo
)

# Simular una trayectoria de descenso
path_x <- c(1.5, 1.5, 1.6, 0.6, -0.2)
path_y <- c(0.35, -0.5, -1.5, -1.5, -1.6)
path_z <- mapply(z_func, path_x, path_y)+0.005

# Superficie con malla
plot_ly(
  x = x,
  y = y,
  z = z,
  type = "surface",
  colorscale = colorscale_custom,
  showscale = FALSE,
  contours = list(
    x = list(show = TRUE, color = "black", width = 1),
    y = list(show = TRUE, color = "black", width = 1)
  )
) |> 
  add_trace(
    x = path_x,
    y = path_y,
    z = path_z,
    type = "scatter3d",
    mode = "lines+markers",
    line = list(color = "black", width = 5),
    marker = list(size = 5, color = "black"),
    showlegend = FALSE) |> 
      
  layout(scene = list(
    camera = list(eye = list(x = 1.2, y = 1.2, z = 0.7)),
    xaxis = list(title = "x", dtick = 1),
    yaxis = list(title = "y", dtick = 1),
    zaxis = list(title = "z", dtick = 1)
  ))

