window.TFM_DATA = {
  overview: [
    { name: "Radiómica · R07", micro: 0.4948, macro: 0.5011, color: "green" },
    { name: "DL · solo imagen", micro: 0.4852, macro: 0.4804, color: "blue" },
    { name: "DL · multimodal", micro: 0.4800, macro: 0.4752, color: "violet" },
    { name: "DL · transferencia", micro: 0.4723, macro: 0.4696, color: "coral" }
  ],
  radiomics: [
    { id: "R07", model: "Gradient Boosting", features: "Basic", data: "Clínica completa", masks: "P+N+V", micro: 0.4948, macro: 0.5011, subset: 0.6000 },
    { id: "R10", model: "Gradient Boosting", features: "Basic", data: "Clínica reducida", masks: "P+N+V", micro: 0.4767, macro: 0.4803, subset: 0.5805 },
    { id: "R04", model: "Extra Trees", features: "Basic", data: "—", masks: "N", micro: 0.4524, macro: 0.4375, subset: 0.6154 },
    { id: "R05", model: "Extra Trees", features: "Basic", data: "—", masks: "N", micro: 0.4524, macro: 0.4375, subset: 0.6154 },
    { id: "R02", model: "Extra Trees", features: "Basic", data: "—", masks: "N", micro: 0.4485, macro: 0.4394, subset: 0.6154 },
    { id: "R11", model: "LightGBM", features: "Basic", data: "—", masks: "V", micro: 0.4468, macro: 0.4301, subset: 0.5854 },
    { id: "R17", model: "Extra Trees", features: "Basic", data: "—", masks: "N", micro: 0.4431, macro: 0.4340, subset: 0.6051 },
    { id: "R25", model: "LightGBM", features: "Basic", data: "Clínica + geometría", masks: "N+V", micro: 0.4420, macro: 0.4125, subset: 0.5805 },
    { id: "R01", model: "XGBoost", features: "Extended", data: "—", masks: "P+N+B", micro: 0.4405, macro: 0.4216, subset: 0.6049 },
    { id: "R26", model: "XGBoost", features: "Extended", data: "Clínica reducida", masks: "P+N+V", micro: 0.4379, macro: 0.4288, subset: 0.5805 }
  ],
  deepLearning: {
    image: {
      title: "Mejor configuración por arquitectura",
      note: "Resultados con umbral fijo y umbrales seleccionados en validación interna.",
      columns: ["Modelo", "Preprocesado", "Entrada", "F1 micro 0,5", "F1 macro 0,5", "F1 micro ajustado", "F1 macro ajustado"],
      rows: [
        ["ResNet-18", "128×256×256 · sin ventana HU", "TC restringida al nódulo", 0.4817, 0.4798, 0.4461, 0.4477],
        ["ResNet-34", "128×256×256 · ventanas separadas", "TC restringida al nódulo", 0.4802, 0.4776, 0.4762, 0.4747],
        ["ResNet-10", "128×256×256 · ventanas separadas", "TC pulmonar + M(P,N,V)", 0.4798, 0.4731, 0.4503, 0.4474],
        ["SE-ResNet-50", "128³ · ventanas separadas", "TC restringida al nódulo", 0.4688, 0.4706, 0.4658, 0.4687],
        ["DenseNet-121", "128×256×256 · HU [−1350,150]", "TC pulmonar + M(P,N)", 0.4645, 0.4562, 0.4354, 0.4284]
      ]
    },
    multimodal: {
      title: "Configuraciones multimodales destacadas",
      note: "La selección multimodal es exploratoria porque parte de configuraciones de imagen priorizadas previamente.",
      columns: ["Modelo", "Preprocesado", "Entrada", "F1 micro 0,5", "F1 macro 0,5", "F1 micro ajustado", "F1 macro ajustado"],
      rows: [
        ["ResNet-34", "128×256×256 · sin ventana HU", "TC restringida al nódulo", 0.4800, 0.4752, 0.4522, 0.4519],
        ["ResNet-34", "128×256×256 · ventanas separadas", "TC restringida al nódulo", 0.4789, 0.4702, 0.4776, 0.4683],
        ["ResNet-18", "128×256×256 · sin ventana HU", "TC restringida al nódulo", 0.4703, 0.4674, 0.4727, 0.4643],
        ["ResNet-10", "128×256×256 · HU [−1350,150]", "TC restringida al nódulo", 0.4496, 0.4479, 0.4591, 0.4580],
        ["ResNet-10", "128×256×256 · sin ventana HU", "TC pulmonar + M(P,N)", 0.4457, 0.4446, 0.4570, 0.4513]
      ]
    },
    transfer: {
      title: "Transferencia y ajuste fino",
      note: "Comparación de inicializaciones MedicalNet, Models Genesis y LUNA16.",
      columns: ["Estrategia", "Inicialización", "Modelo", "Entrada", "F1 micro 0,5", "F1 macro 0,5", "F1 micro ajustado"],
      rows: [
        ["Directa", "MedicalNet", "ResNet-18", "TC pulmonar + M(P,N,V)", 0.4164, 0.4129, 0.4282],
        ["Directa", "Models Genesis", "Codificador 3D", "TC pulmonar + M(P,N,V)", 0.3710, 0.3707, 0.4389],
        ["Freeze/unfreeze", "MedicalNet", "ResNet-34", "TC restringida al nódulo · ventanas", 0.4323, 0.4220, 0.4466],
        ["Freeze/unfreeze", "Models Genesis", "Codificador 3D", "TC restringida al nódulo · ventanas", 0.4723, 0.4696, 0.4715],
        ["Freeze/unfreeze", "LUNA16", "ResNet-34", "TC restringida al nódulo", 0.4499, 0.4418, 0.4505]
      ]
    }
  },
  pu: {
    hemorrhage: [
      ["Clínica + geometría", "Supervisado", 0.422, 0.646, 0.431, 0.458, 0.796, 0.604, 0.245, 0.197],
      ["Clínica + geometría", "Etiq. neg. ingenuo", 0.352, 0.630, 0.403, 0.521, 0.675, 0.593, 0.172, 0.223],
      ["Clínica + geometría", "Elkan-Noto", 0.370, 0.673, 0.478, 0.562, 0.758, 0.653, 0.292, 0.493],
      ["Clínica + geometría", "Bagging PU", 0.372, 0.663, 0.457, 0.604, 0.682, 0.642, 0.249, 0.236],
      ["Radiómica", "Supervisado", 0.416, 0.654, 0.426, 0.542, 0.694, 0.613, 0.208, 0.174],
      ["Radiómica", "Etiq. neg. ingenuo", 0.370, 0.623, 0.404, 0.438, 0.777, 0.583, 0.204, 0.200],
      ["Radiómica", "Elkan-Noto", 0.309, 0.588, 0.423, 0.604, 0.618, 0.611, 0.190, 0.365],
      ["Radiómica", "Bagging PU", 0.396, 0.645, 0.452, 0.583, 0.694, 0.636, 0.243, 0.238],
      ["Combinada", "Supervisado", 0.428, 0.657, 0.462, 0.625, 0.669, 0.647, 0.254, 0.174],
      ["Combinada", "Etiq. neg. ingenuo", 0.379, 0.637, 0.439, 0.521, 0.739, 0.620, 0.235, 0.204],
      ["Combinada", "Elkan-Noto", 0.294, 0.593, 0.408, 0.604, 0.586, 0.595, 0.162, 0.467],
      ["Combinada", "Bagging PU", 0.445, 0.671, 0.450, 0.604, 0.669, 0.636, 0.236, 0.237]
    ],
    pneumothorax: [
      ["Clínica + geometría", "Supervisado", 0.340, 0.532, 0.412, 0.524, 0.549, 0.536, 0.068, 0.240],
      ["Clínica + geometría", "Etiq. neg. ingenuo", 0.387, 0.593, 0.524, 0.794, 0.451, 0.598, 0.233, 0.250],
      ["Clínica + geometría", "Elkan-Noto", 0.390, 0.624, 0.487, 0.603, 0.613, 0.608, 0.200, 0.504],
      ["Clínica + geometría", "Bagging PU", 0.369, 0.560, 0.423, 0.524, 0.577, 0.550, 0.094, 0.246],
      ["Radiómica", "Supervisado", 0.404, 0.587, 0.419, 0.429, 0.725, 0.558, 0.152, 0.242],
      ["Radiómica", "Etiq. neg. ingenuo", 0.398, 0.594, 0.466, 0.492, 0.725, 0.597, 0.212, 0.263],
      ["Radiómica", "Elkan-Noto", 0.350, 0.551, 0.400, 0.429, 0.683, 0.541, 0.108, 0.422],
      ["Radiómica", "Bagging PU", 0.404, 0.577, 0.406, 0.429, 0.697, 0.547, 0.122, 0.273],
      ["Combinada", "Supervisado", 0.374, 0.557, 0.412, 0.444, 0.683, 0.551, 0.123, 0.252],
      ["Combinada", "Etiq. neg. ingenuo", 0.410, 0.591, 0.438, 0.476, 0.690, 0.573, 0.160, 0.252],
      ["Combinada", "Elkan-Noto", 0.345, 0.541, 0.367, 0.429, 0.599, 0.506, 0.025, 0.435],
      ["Combinada", "Bagging PU", 0.386, 0.587, 0.460, 0.460, 0.761, 0.592, 0.221, 0.261]
    ]
  },
  pu3d: [
    { target: "Hemorragia", method: "Supervisado", ap: 0.287, f1: 0.294 },
    { target: "Hemorragia", method: "Elkan-Noto", ap: 0.241, f1: 0.291 },
    { target: "Hemorragia", method: "nnPU", ap: 0.272, f1: 0.362 },
    { target: "Neumotórax", method: "Supervisado", ap: 0.416, f1: 0.357 },
    { target: "Neumotórax", method: "Elkan-Noto", ap: 0.378, f1: 0.447 },
    { target: "Neumotórax", method: "nnPU", ap: 0.381, f1: 0.432 }
  ],
  rules: [
    { set: "Clínica", target: "Hemorragia", rule: "Edad > 60,0", consensus: "Moderado", configs: "16/24", wracc: 0.0086, delta: 0.0085 },
    { set: "Clínica + geometría", target: "Hemorragia", rule: "Tamaño del nódulo ≤ 31,05 mm", consensus: "Moderado", configs: "16/24", wracc: 0.0257, delta: 0.0638 },
    { set: "Clínica", target: "Neumotórax", rule: "Antecedente de tabaquismo", consensus: "Moderado", configs: "16/24", wracc: 0.0084, delta: 0.0111 },
    { set: "Clínica + geometría", target: "Neumotórax", rule: "Profundidad pleural ≤ 10,21 mm y tamaño ≤ 42,80 mm", consensus: "Moderado", configs: "12/24", wracc: 0.0435, delta: 0.0690 },
    { set: "Clínica", target: "Sin complicación", rule: "Sin patología pulmonar", consensus: "Fuerte", configs: "24/24", wracc: 0.0511, delta: 0.0816 },
    { set: "Clínica + geometría", target: "Sin complicación", rule: "Tamaño del nódulo > 42,50 mm", consensus: "Fuerte", configs: "24/24", wracc: 0.0545, delta: 0.1671 }
  ],
  subgroupModels: [
    { subgroup: "Edad > 60", model: "Regresión logística", general: 0.420, specialized: 0.471 },
    { subgroup: "Tamaño ≤ 31,05 mm", model: "Regresión logística", general: 0.491, specialized: 0.549 },
    { subgroup: "Tabaquismo", model: "Random Forest", general: 0.448, specialized: 0.480 },
    { subgroup: "Profundidad ≤ 10,21 mm y tamaño ≤ 42,80 mm", model: "Extra Trees", general: 0.509, specialized: 0.549 }
  ]
};
