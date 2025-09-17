import './style.css'
import * as THREE from 'three'

const app = document.querySelector<HTMLDivElement>('#app')!

const scene = new THREE.Scene()
// CPPN Neural Network Shader - from neural-network-hero.tsx
const vertexShader = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const fragmentShader = `
  #ifdef GL_ES
    precision lowp float;
  #endif
  uniform float iTime;
  uniform vec2 iResolution;
  varying vec2 vUv;

  vec4 buf[8];

  vec4 sigmoid(vec4 x) { return 1. / (1. + exp(-x)); }

  vec4 cppn_fn(vec2 coordinate, float in0, float in1, float in2) {
    // layer 1 *********************************************************************
    buf[6] = vec4(coordinate.x, coordinate.y, 0.3948333106474662 + in0, 0.36 + in1);
    buf[7] = vec4(0.14 + in2, sqrt(coordinate.x * coordinate.x + coordinate.y * coordinate.y), 0., 0.);

    // layer 2 ********************************************************************
    buf[0] = mat4(vec4(6.5404263, -3.6126034, 0.7590882, -1.13613), vec4(2.4582713, 3.1660357, 1.2219609, 0.06276096), vec4(-5.478085, -6.159632, 1.8701609, -4.7742867), vec4(6.039214, -5.542865, -0.90925294, 3.251348))
    * buf[6]
    + mat4(vec4(0.8473259, -5.722911, 3.975766, 1.6522468), vec4(-0.24321538, 0.5839259, -1.7661959, -5.350116), vec4(0.0, 0.0, 0.0, 0.0), vec4(0.0, 0.0, 0.0, 0.0))
    * buf[7]
    + vec4(0.21808943, 1.1243913, -1.7969975, 5.0294676);

    buf[1] = mat4(vec4(-3.3522482, -6.0612736, 0.55641043, -4.4719114), vec4(0.8631464, 1.7432913, 5.643898, 1.6106541), vec4(2.4941394, -3.5012043, 1.7184316, 6.357333), vec4(3.310376, 8.209261, 1.1355612, -1.165539))
    * buf[6]
    + mat4(vec4(5.24046, -13.034365, 0.009859298, 15.870829), vec4(2.987511, 3.129433, -0.89023495, -1.6822904), vec4(0.0, 0.0, 0.0, 0.0), vec4(0.0, 0.0, 0.0, 0.0))
    * buf[7]
    + vec4(-5.9457836, -6.573602, -0.8812491, 1.5436668);

    buf[0] = sigmoid(buf[0]);
    buf[1] = sigmoid(buf[1]);

    // layer 3 ********************************************************************
    buf[2] = mat4(vec4(-15.219568, 8.095543, -2.429353, -1.9381982), vec4(-5.951362, 4.3115187, 2.6393783, 1.274315), vec4(-7.3145227, 6.7297835, 5.2473326, 5.9411426), vec4(5.0796127, 8.979051, -1.7278991, -1.158976))
    * buf[6]
    + mat4(vec4(-11.967154, -11.608155, 6.1486754, 11.237008), vec4(2.124141, -6.263192, -1.7050359, -0.7021966), vec4(0.0, 0.0, 0.0, 0.0), vec4(0.0, 0.0, 0.0, 0.0))
    * buf[7]
    + vec4(-4.17164, -3.2281182, -4.576417, -3.6401186);

    buf[3] = mat4(vec4(3.1832156, -13.738922, 1.879223, 3.233465), vec4(0.64300746, 12.768129, 1.9141049, 0.50990224), vec4(-0.049295485, 4.4807224, 1.4733979, 1.801449), vec4(5.0039253, 13.000481, 3.3991797, -4.5561905))
    * buf[6]
    + mat4(vec4(-0.1285731, 7.720628, -3.1425676, 4.742367), vec4(0.6393625, 3.714393, -0.8108378, -0.39174938), vec4(0.0, 0.0, 0.0, 0.0), vec4(0.0, 0.0, 0.0, 0.0))
    * buf[7]
    + vec4(-1.1811101, -21.621881, 0.7851888, 1.2329718);

    buf[2] = sigmoid(buf[2]);
    buf[3] = sigmoid(buf[3]);

    // layer 5 & 6 ****************************************************************
    buf[4] = mat4(vec4(5.214916, -7.183024, 2.7228765, 2.6592617), vec4(-5.601878, -25.3591, 4.067988, 0.4602802), vec4(-10.57759, 24.286327, 21.102104, 37.546658), vec4(4.3024497, -1.9625226, 2.3458803, -1.372816))
    * buf[0]
    + mat4(vec4(-17.6526, -10.507558, 2.2587414, 12.462782), vec4(6.265566, -502.75443, -12.642513, 0.9112289), vec4(-10.983244, 20.741234, -9.701768, -0.7635988), vec4(5.383626, 1.4819539, -4.1911616, -4.8444734))
    * buf[1]
    + mat4(vec4(12.785233, -16.345072, -0.39901125, 1.7955981), vec4(-30.48365, -1.8345358, 1.4542528, -1.1118771), vec4(19.872723, -7.337935, -42.941723, -98.52709), vec4(8.337645, -2.7312303, -2.2927687, -36.142323))
    * buf[2]
    + mat4(vec4(-16.298317, 3.5471997, -0.44300047, -9.444417), vec4(57.5077, -35.609753, 16.163465, -4.1534753), vec4(-0.07470326, -3.8656476, -7.0901804, 3.1523974), vec4(-12.559385, -7.077619, 1.490437, -0.8211543))
    * buf[3]
    + vec4(-7.67914, 15.927437, 1.3207729, -1.6686112);

    buf[5] = mat4(vec4(-1.4109162, -0.372762, -3.770383, -21.367174), vec4(-6.2103205, -9.35908, 0.92529047, 8.82561), vec4(11.460242, -22.348068, 13.625772, -18.693201), vec4(-0.3429052, -3.9905605, -2.4626114, -0.45033523))
    * buf[0]
    + mat4(vec4(7.3481627, -4.3661838, -6.3037653, -3.868115), vec4(1.5462853, 6.5488915, 1.9701879, -0.58291394), vec4(6.5858274, -2.2180402, 3.7127688, -1.3730392), vec4(-5.7973905, 10.134961, -2.3395722, -5.965605))
    * buf[1]
    + mat4(vec4(-2.5132585, -6.6685553, -1.4029363, -0.16285264), vec4(-0.37908727, 0.53738135, 4.389061, -1.3024765), vec4(-0.70647055, 2.0111287, -5.1659346, -3.728635), vec4(-13.562562, 10.487719, -0.9173751, -2.6487076))
    * buf[2]
    + mat4(vec4(-8.645013, 6.5546675, -6.3944063, -5.5933375), vec4(-0.57783127, -1.077275, 36.91025, 5.736769), vec4(14.283112, 3.7146652, 7.1452246, -4.5958776), vec4(2.7192075, 3.6021907, -4.366337, -2.3653464))
    * buf[3]
    + vec4(-5.9000807, -4.329569, 1.2427121, 8.59503);

    buf[4] = sigmoid(buf[4]);
    buf[5] = sigmoid(buf[5]);

    // layer 7 & 8 ****************************************************************
    buf[6] = mat4(vec4(-1.61102, 0.7970257, 1.4675229, 0.20917463), vec4(-28.793737, -7.1390953, 1.5025433, 4.656581), vec4(-10.94861, 39.66238, 0.74318546, -10.095605), vec4(-0.7229728, -1.5483948, 0.7301322, 2.1687684))
    * buf[0]
    + mat4(vec4(3.2547753, 21.489103, -1.0194173, -3.3100595), vec4(-3.7316632, -3.3792162, -7.223193, -0.23685838), vec4(13.1804495, 0.7916005, 5.338587, 5.687114), vec4(-4.167605, -17.798311, -6.815736, -1.6451967))
    * buf[1]
    + mat4(vec4(0.604885, -7.800309, -7.213122, -2.741014), vec4(-3.522382, -0.12359311, -0.5258442, 0.43852118), vec4(9.6752825, -22.853785, 2.062431, 0.099892326), vec4(-4.3196306, -17.730087, 2.5184598, 5.30267))
    * buf[2]
    + mat4(vec4(-6.545563, -15.790176, -6.0438633, -5.415399), vec4(-43.591583, 28.551912, -16.00161, 18.84728), vec4(4.212382, 8.394307, 3.0958717, 8.657522), vec4(-5.0237565, -4.450633, -4.4768, -5.5010443))
    * buf[3]
    + mat4(vec4(1.6985557, -67.05806, 6.897715, 1.9004834), vec4(1.8680354, 2.3915145, 2.5231109, 4.081538), vec4(11.158006, 1.7294737, 2.0738268, 7.386411), vec4(-4.256034, -306.24686, 8.258898, -17.132736))
    * buf[4]
    + mat4(vec4(1.6889864, -4.5852966, 3.8534803, -6.3482175), vec4(1.3543309, -1.2640043, 9.932754, 2.9079645), vec4(-5.2770967, 0.07150358, -0.13962056, 3.3269649), vec4(28.34703, -4.918278, 6.1044083, 4.085355))
    * buf[5]
    + vec4(6.6818056, 12.522166, -3.7075126, -4.104386);

    buf[7] = mat4(vec4(-8.265602, -4.7027016, 5.098234, 0.7509808), vec4(8.6507845, -17.15949, 16.51939, -8.884479), vec4(-4.036479, -2.3946867, -2.6055532, -1.9866527), vec4(-2.2167742, -1.8135649, -5.9759874, 4.8846445))
    * buf[0]
    + mat4(vec4(6.7790847, 3.5076547, -2.8191125, -2.7028968), vec4(-5.743024, -0.27844876, 1.4958696, -5.0517144), vec4(13.122226, 15.735168, -2.9397483, -4.101023), vec4(-14.375265, -5.030483, -6.2599335, 2.9848232))
    * buf[1]
    + mat4(vec4(4.0950394, -0.94011575, -5.674733, 4.755022), vec4(4.3809423, 4.8310084, 1.7425908, -3.437416), vec4(2.117492, 0.16342592, -104.56341, 16.949184), vec4(-5.22543, -2.994248, 3.8350096, -1.9364246))
    * buf[2]
    + mat4(vec4(-5.900337, 1.7946124, -13.604192, -3.8060522), vec4(6.6583457, 31.911177, 25.164474, 91.81147), vec4(11.840538, 4.1503043, -0.7314397, 6.768467), vec4(-6.3967767, 4.034772, 6.1714606, -0.32874924))
    * buf[3]
    + mat4(vec4(3.4992442, -196.91893, -8.923708, 2.8142626), vec4(3.4806502, -3.1846354, 5.1725626, 5.1804223), vec4(-2.4009497, 15.585794, 1.2863957, 2.0252278), vec4(-71.25271, -62.441242, -8.138444, 0.50670296))
    * buf[4]
    + mat4(vec4(-12.291733, -11.176166, -7.3474145, 4.390294), vec4(10.805477, 5.6337385, -0.9385842, -4.7348723), vec4(-12.869276, -7.039391, 5.3029537, 7.5436664), vec4(1.4593618, 8.91898, 3.5101583, 5.840625))
    * buf[5]
    + vec4(2.2415268, -6.705987, -0.98861027, -2.117676);

    buf[6] = sigmoid(buf[6]);
    buf[7] = sigmoid(buf[7]);

    // layer 9 ********************************************************************
    buf[0] = mat4(vec4(1.6794263, 1.3817469, 2.9625452, 0.0), vec4(-1.8834411, -1.4806935, -3.5924516, 0.0), vec4(-1.3279216, -1.0918057, -2.3124623, 0.0), vec4(0.2662234, 0.23235129, 0.44178495, 0.0))
    * buf[0]
    + mat4(vec4(-0.6299101, -0.5945583, -0.9125601, 0.0), vec4(0.17828953, 0.18300213, 0.18182953, 0.0), vec4(-2.96544, -2.5819945, -4.9001055, 0.0), vec4(1.4195864, 1.1868085, 2.5176322, 0.0))
    * buf[1]
    + mat4(vec4(-1.2584374, -1.0552157, -2.1688404, 0.0), vec4(-0.7200217, -0.52666044, -1.438251, 0.0), vec4(0.15345335, 0.15196142, 0.272854, 0.0), vec4(0.945728, 0.8861938, 1.2766753, 0.0))
    * buf[2]
    + mat4(vec4(-2.4218085, -1.968602, -4.35166, 0.0), vec4(-22.683098, -18.0544, -41.954372, 0.0), vec4(0.63792, 0.5470648, 1.1078634, 0.0), vec4(-1.5489894, -1.3075932, -2.6444845, 0.0))
    * buf[3]
    + mat4(vec4(-0.49252132, -0.39877754, -0.91366625, 0.0), vec4(0.95609266, 0.7923952, 1.640221, 0.0), vec4(0.30616966, 0.15693925, 0.8639857, 0.0), vec4(1.1825981, 0.94504964, 2.176963, 0.0))
    * buf[4]
    + mat4(vec4(0.35446745, 0.3293795, 0.59547555, 0.0), vec4(-0.58784515, -0.48177817, -1.0614829, 0.0), vec4(2.5271258, 1.9991658, 4.6846647, 0.0), vec4(0.13042648, 0.08864098, 0.30187556, 0.0))
    * buf[5]
    + mat4(vec4(-1.7718065, -1.4033192, -3.3355875, 0.0), vec4(3.1664357, 2.638297, 5.378702, 0.0), vec4(-3.1724713, -2.6107926, -5.549295, 0.0), vec4(-2.851368, -2.249092, -5.3013067, 0.0))
    * buf[6]
    + mat4(vec4(1.5203838, 1.2212278, 2.8404984, 0.0), vec4(1.5210563, 1.2651345, 2.683903, 0.0), vec4(2.9789467, 2.4364579, 5.2347264, 0.0), vec4(2.2270417, 1.8825914, 3.8028636, 0.0))
    * buf[7]
    + vec4(-1.5468478, -3.6171484, 0.24762098, 0.0);

    buf[0] = sigmoid(buf[0]);
    return vec4(buf[0].x , buf[0].y , buf[0].z, 1.0);
  }

  void main() {
    // Make the pattern larger by scaling UV coordinates
    vec2 uv = (vUv * 1.5 - 0.75) * 2.0; uv.y *= -1.0;

    // Speed up the animation with much faster time multipliers
    vec4 originalColor = cppn_fn(uv, 0.2 * sin(2.0 * iTime), 0.2 * sin(2.5 * iTime), 0.2 * sin(2.2 * iTime));

    // Convert to dull teal color palette
    float intensity = (originalColor.r + originalColor.g + originalColor.b) / 3.0;
    vec3 dullTeal = vec3(0.0, 0.4 * intensity, 0.35 * intensity);

    gl_FragColor = vec4(dullTeal, 1.0);
  }
`;

// Create shader material for neural network background
const neuralMaterial = new THREE.ShaderMaterial({
  uniforms: {
    iTime: { value: 0 },
    iResolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) }
  },
  vertexShader,
  fragmentShader,
  side: THREE.DoubleSide
})

// Create background plane for neural shader - make it much larger to cover full screen
const neuralGeometry = new THREE.PlaneGeometry(600, 600)
const neuralPlane = new THREE.Mesh(neuralGeometry, neuralMaterial)
neuralPlane.position.set(0, -80, -50) // Position it much lower on the page
scene.add(neuralPlane)

function updateNeuralBackground(time: number) {
  // Update shader uniforms
  neuralMaterial.uniforms.iTime.value = time
  neuralMaterial.uniforms.iResolution.value.set(window.innerWidth, window.innerHeight)
}

// Initialize with black background
scene.background = new THREE.Color(0x000000)

const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
)
// Position camera to look straight at the grid initially
camera.position.set(0, 0, 50)
camera.lookAt(0, 0, 0)

const renderer = new THREE.WebGLRenderer({ antialias: true })
renderer.setSize(window.innerWidth, window.innerHeight)
renderer.setPixelRatio(window.devicePixelRatio)
app.appendChild(renderer.domElement)

const gridSize = 141 // Increased by 100% (141^2 ≈ 2x more particles than 100^2)
const gridSpacing = 1.42 // Reduced proportionally (2 * 100/141) to maintain same mountain size
const particleCount = gridSize * gridSize

// Calculate screen dimensions for proper 2D spread
const aspect = window.innerWidth / window.innerHeight
const fov = camera.fov * Math.PI / 180
const distance = camera.position.z
const screenHeight = 2 * Math.tan(fov / 2) * distance
const screenWidth = screenHeight * aspect

const geometry = new THREE.BufferGeometry()
const positions = new Float32Array(particleCount * 3)
const originalPositions = new Float32Array(particleCount * 3)
const mountainPositions = new Float32Array(particleCount * 3)
const colors = new Float32Array(particleCount * 3)
const randomAmplitudes = new Float32Array(particleCount) // Store random amplitude for each particle

let index = 0
for (let i = 0; i < gridSize; i++) {
  for (let j = 0; j < gridSize; j++) {
    // Create 2D grid spread much wider - particles overflow beyond screen edges
    const x = (i / (gridSize - 1) - 0.5) * screenWidth * 3.0
    const y = (j / (gridSize - 1) - 0.5) * screenHeight * 3.0
    const z = 0 // All particles at exactly z=0 for true 2D

    // Store initial 2D grid positions
    positions[index * 3] = x
    positions[index * 3 + 1] = y
    positions[index * 3 + 2] = z

    originalPositions[index * 3] = x
    originalPositions[index * 3 + 1] = y
    originalPositions[index * 3 + 2] = z

    // Calculate mountain positions - EXACTLY as in original code
    const terrainX = (i - gridSize / 2) * gridSpacing
    const terrainZ = (j - gridSize / 2) * gridSpacing
    const distanceFromCenter = Math.sqrt(terrainX * terrainX + terrainZ * terrainZ)
    const mountainHeight = Math.max(0, 25 - distanceFromCenter * 0.5) +
                          Math.sin(terrainX * 0.1) * 5 +
                          Math.cos(terrainZ * 0.1) * 5 +
                          Math.random() * 2 // Restored original height

    mountainPositions[index * 3] = terrainX
    mountainPositions[index * 3 + 1] = mountainHeight
    mountainPositions[index * 3 + 2] = terrainZ

    // Randomly choose between teal and white for each particle
    const useWhite = Math.random() > 0.5

    if (useWhite) {
      // White color
      colors[index * 3] = 1.0
      colors[index * 3 + 1] = 1.0
      colors[index * 3 + 2] = 1.0
    } else {
      // Teal color (cyan-like)
      colors[index * 3] = 0.0
      colors[index * 3 + 1] = 0.8
      colors[index * 3 + 2] = 0.8
    }

    // Generate random amplitude for each particle (between 0.2 and 1.5)
    randomAmplitudes[index] = 0.2 + Math.random() * 1.3

    index++
  }
}

geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))

const material = new THREE.PointsMaterial({
  size: 0.15, // Further reduced particle size
  vertexColors: true,
  blending: THREE.AdditiveBlending,
  transparent: true,
  opacity: 0.8
})

const particles = new THREE.Points(geometry, material)
scene.add(particles)

const ambientLight = new THREE.AmbientLight(0xffffff, 0.3)
scene.add(ambientLight)

let isMountain = false
let transitionProgress = 0
let targetProgress = 0
let autoDemo = true
let demoCompleted = false

// Create wrapper for gradient with hover detection
const gradientWrapper = document.createElement('div')
gradientWrapper.style.position = 'fixed'
gradientWrapper.style.bottom = '120px'
gradientWrapper.style.top = 'auto'
gradientWrapper.style.left = '50%'
gradientWrapper.style.transform = 'translateX(-50%)'
gradientWrapper.style.width = '170px'
gradientWrapper.style.height = '170px'
gradientWrapper.style.borderRadius = '50%'
gradientWrapper.style.zIndex = '998'

// Create enhanced shadow element
const shadowElement = document.createElement('div')
shadowElement.style.position = 'absolute'
shadowElement.style.width = '100%'
shadowElement.style.height = '100%'
shadowElement.style.borderRadius = '50%'
shadowElement.style.background = `
  radial-gradient(circle at 45% 45%,
    rgba(0, 0, 0, 0.9) 0%,
    rgba(0, 0, 0, 0.6) 15%,
    transparent 30%
  ),
  radial-gradient(circle at 25% 65%,
    rgba(0, 0, 0, 0.8) 0%,
    rgba(0, 0, 0, 0.4) 10%,
    transparent 25%
  ),
  radial-gradient(circle at 75% 35%,
    rgba(0, 0, 0, 0.7) 0%,
    transparent 20%
  ),
  radial-gradient(circle at 30% 30%,
    rgba(0, 255, 220, 1.0) 0%,
    rgba(255, 120, 0, 1.0) 30%,
    rgba(0, 255, 220, 0.8) 60%,
    transparent 100%
  ),
  radial-gradient(circle at 70% 70%,
    rgba(255, 120, 0, 1.0) 0%,
    rgba(0, 255, 220, 0.9) 30%,
    transparent 70%
  )
`
shadowElement.style.backgroundSize = '400% 400%'
shadowElement.style.animation = 'gradientMove 8s ease-in-out infinite'
shadowElement.id = 'gradientElement'
// Dynamic shadow will be set by animation function
shadowElement.style.filter = 'blur(15px)'
shadowElement.style.transition = 'mask 0.1s ease, -webkit-mask 0.1s ease'
shadowElement.style.zIndex = '999'
shadowElement.style.pointerEvents = 'all'

// Create hover black spot element - matching the gradient's black spots
const hoverSpot = document.createElement('div')
hoverSpot.style.position = 'fixed'
hoverSpot.style.width = '170px'  // Same size as gradient circle
hoverSpot.style.height = '170px'
hoverSpot.style.borderRadius = '50%'
hoverSpot.style.pointerEvents = 'none'
hoverSpot.style.zIndex = '1000'
hoverSpot.style.opacity = '0'
hoverSpot.style.transition = 'opacity 0.3s ease'
hoverSpot.style.transform = 'translate(-50%, -50%)'
hoverSpot.style.filter = 'blur(15px)'  // Same blur as main gradient

// Test with document-wide mouse tracking
document.addEventListener('mousemove', (e) => {
  const rect = gradientWrapper.getBoundingClientRect()
  const centerX = rect.left + rect.width / 2
  const centerY = rect.top + rect.height / 2
  const distance = Math.sqrt(Math.pow(e.clientX - centerX, 2) + Math.pow(e.clientY - centerY, 2))

  if (distance < rect.width / 2) {
    // Mouse is within the circle - calculate relative position
    const relX = ((e.clientX - rect.left) / rect.width) * 100
    const relY = ((e.clientY - rect.top) / rect.height) * 100

    // Update hover spot background with very large black gradient at mouse position
    hoverSpot.style.background = `
      radial-gradient(circle at ${relX}% ${relY}%,
        rgba(0, 0, 0, 0.95) 0%,
        rgba(0, 0, 0, 0.8) 40%,
        rgba(0, 0, 0, 0.5) 60%,
        rgba(0, 0, 0, 0.2) 75%,
        transparent 85%
      )
    `

    hoverSpot.style.left = centerX + 'px'
    hoverSpot.style.top = centerY + 'px'
    hoverSpot.style.opacity = '1'
  } else {
    // Mouse is outside the circle
    hoverSpot.style.opacity = '0'
  }
})

// Append shadow element to wrapper
gradientWrapper.appendChild(shadowElement)

// Create mouse hover effect element
const hoverEffect = document.createElement('div')
hoverEffect.style.position = 'fixed'
hoverEffect.style.width = '180px'
hoverEffect.style.height = '180px'
hoverEffect.style.borderRadius = '50%'
hoverEffect.style.pointerEvents = 'none'
hoverEffect.style.zIndex = '1001'
hoverEffect.style.opacity = '0'
hoverEffect.style.transition = 'opacity 0.3s ease'
hoverEffect.style.mixBlendMode = 'normal'
hoverEffect.style.background = `
  radial-gradient(circle,
    transparent 0%,
    transparent 60%,
    rgba(51, 229, 204, 0.8) 70%,
    rgba(255, 153, 51, 1.0) 100%
  )
`
hoverEffect.style.mask = `
  radial-gradient(circle,
    black 0%,
    black 60%,
    transparent 100%
  )
`
hoverEffect.style.webkitMask = `
  radial-gradient(circle,
    black 0%,
    black 60%,
    transparent 100%
  )
`
hoverEffect.style.filter = 'blur(8px)'
document.body.appendChild(hoverEffect)


// Create button wrapper matching reference design
const buttonWrapper = document.createElement('div')
buttonWrapper.className = 'button-wrapper'
buttonWrapper.style.position = 'fixed'
buttonWrapper.style.setProperty('bottom', '120px', 'important')
buttonWrapper.style.setProperty('top', 'auto', 'important')
buttonWrapper.style.left = '50%'
buttonWrapper.style.transform = 'translate(-50%, 0)'
buttonWrapper.style.zIndex = '1000'
buttonWrapper.style.width = 'auto'
buttonWrapper.style.height = 'auto'
console.log('Button wrapper positioned at bottom:', buttonWrapper.style.bottom)

// Create button container with new gradient styling
const buttonContainer = document.createElement('div')
buttonContainer.className = 'gradient-button'
buttonContainer.style.position = 'relative'
buttonContainer.style.width = '200px'
buttonContainer.style.height = '200px'
buttonContainer.style.cursor = 'pointer'
buttonContainer.style.background = 'transparent'
buttonContainer.style.borderRadius = '50%'
buttonContainer.style.border = 'none'
buttonContainer.style.backdropFilter = 'none'
buttonContainer.style.boxShadow = 'none'
buttonContainer.style.transition = 'transform 0.3s ease, box-shadow 0.3s ease'

// Remove color streams - keep only gradient

// Create button element with WebGL-like glass morphism
const button = document.createElement('button')
button.setAttribute('data-v-79b836d4', '')
button.setAttribute('main', 'true')
button.setAttribute('aria-label', 'Audio/Start Button')
button.className = 'webgl-bg'
button.style.position = 'absolute'
button.style.top = '25%'
button.style.left = '18%'
button.style.marginLeft = '-50px'
button.style.marginTop = '-25px'
button.style.width = '100px'
button.style.height = '50px'
button.style.transform = 'none'
button.style.transformOrigin = 'center center'
button.style.background = 'transparent'
button.style.border = 'none'
button.style.display = 'flex'
button.style.alignItems = 'center'
button.style.justifyContent = 'center'
button.style.width = 'auto'
button.style.height = 'auto'
button.style.padding = '60px 80px'
button.style.color = 'white'
button.style.fontSize = '32px'
button.style.fontWeight = '600'
button.style.fontFamily = 'system-ui, -apple-system, sans-serif'
button.style.cursor = 'pointer'
button.style.zIndex = '10'
button.style.textShadow = '0 0 20px rgba(255, 255, 255, 0.5)'
button.style.backdropFilter = 'none'
button.style.transition = 'all 0.3s ease'
button.style.pointerEvents = 'none'
button.style.boxShadow = 'none'

const span = document.createElement('span')
span.setAttribute('data-v-79b836d4', '')
span.textContent = 'Start'
span.style.pointerEvents = 'none'
button.appendChild(span)

// Add styling with animated gradient
const style = document.createElement('style')
style.textContent = `
  @keyframes gradientMove {
    0% {
      background-position: 30% 30%, 70% 70%;
    }
    25% {
      background-position: 60% 20%, 40% 80%;
    }
    50% {
      background-position: 80% 60%, 20% 40%;
    }
    75% {
      background-position: 40% 80%, 60% 20%;
    }
    100% {
      background-position: 30% 30%, 70% 70%;
    }
  }

  .clicked {
    transform: scale(1.1) !important;
    transform-origin: center center !important;
    text-shadow: 0 0 40px rgba(255, 255, 255, 0.8) !important;
  }

  /* Force button to bottom */
  .button-wrapper {
    position: fixed !important;
    bottom: 120px !important;
    top: auto !important;
    left: 50% !important;
    transform: translate(-50%, 0) !important;
    z-index: 1000 !important;
  }

  /* Responsive adjustments */
  @media (max-width: 768px) {
    .button-wrapper {
      bottom: 80px !important;
      top: auto !important;
    }
  }

  /* Smooth performance optimizations */
  .button-wrapper, .button-wrapper * {
    will-change: transform, opacity;
    transform-style: preserve-3d;
    backface-visibility: hidden;
  }
`
document.head.appendChild(style)

// Assemble the simplified button structure
buttonContainer.appendChild(button)
buttonWrapper.appendChild(buttonContainer)

// Enhanced interaction handlers with WebGL integration
buttonWrapper.addEventListener('click', () => {
  // Disable auto-demo when user interacts
  autoDemo = false

  isMountain = !isMountain
  targetProgress = isMountain ? 1 : 0
  span.textContent = 'Start'

  // Add subtle feedback animation maintaining position
  button.style.transform = 'scale(0.95)'

  setTimeout(() => {
    button.style.transform = 'scale(1)'
  }, 150)

  // Add particle burst effect at button click
  if (isMountain) {
    // Create temporary sparkle effect
    for (let i = 0; i < 5; i++) {
      setTimeout(() => {
        const sparkle = document.createElement('div')
        sparkle.style.position = 'fixed'
        sparkle.style.top = `${70 + Math.random() * 20}px`
        sparkle.style.left = `${window.innerWidth/2 + (Math.random() - 0.5) * 60}px`
        sparkle.style.width = '3px'
        sparkle.style.height = '3px'
        sparkle.style.background = i % 2 === 0 ? 'rgba(255, 120, 0, 1.0)' : 'rgba(0, 255, 220, 1.0)'
        sparkle.style.borderRadius = '50%'
        sparkle.style.pointerEvents = 'none'
        sparkle.style.zIndex = '1001'
        sparkle.style.boxShadow = `0 0 10px ${i % 2 === 0 ? 'rgba(255, 120, 0, 1.0)' : 'rgba(0, 255, 220, 0.8)'}`

        document.body.appendChild(sparkle)

        // Animate sparkle
        sparkle.animate([
          { transform: 'scale(0) translateY(0px)', opacity: '1' },
          { transform: 'scale(1.5) translateY(-20px)', opacity: '0.5' },
          { transform: 'scale(0) translateY(-40px)', opacity: '0' }
        ], {
          duration: 800,
          easing: 'ease-out'
        }).onfinish = () => sparkle.remove()
      }, i * 100)
    }
  }
})

let isHovered = 0
let hoverTransition = 0


// Show hover effect when over button area - WebGL style
buttonWrapper.addEventListener('mouseenter', () => {
  isHovered = 1
  console.log('Hover entered')
})

buttonWrapper.addEventListener('mouseleave', () => {
  isHovered = 0
  console.log('Hover left')
})

// Add click effects like element.txt
buttonWrapper.addEventListener('click', () => {
  span.classList.add('clicked')
  setTimeout(() => {
    span.classList.remove('clicked')
  }, 300)
})
// Create copyright text element
const copyrightText = document.createElement('div')
copyrightText.textContent = '2025 © VOS9X'
copyrightText.style.position = 'fixed'
copyrightText.style.bottom = '20px'
copyrightText.style.left = '40px'
copyrightText.style.color = '#ffffff'
copyrightText.style.fontSize = '14px'
copyrightText.style.fontWeight = 'bold'
copyrightText.style.fontFamily = 'system-ui, -apple-system, sans-serif'
copyrightText.style.zIndex = '999'
copyrightText.style.pointerEvents = 'none'
copyrightText.style.letterSpacing = '0.5px'

// Create VOS9X description text element
const descriptionText = document.createElement('div')
descriptionText.innerHTML = `VOS9X™ is a multi-asset<br>investment and growth<br>equity firm`
descriptionText.style.position = 'fixed'
descriptionText.style.top = '40px'
descriptionText.style.right = '40px'
descriptionText.style.color = '#ffffff'
descriptionText.style.fontSize = '16px'
descriptionText.style.fontWeight = '300'
descriptionText.style.fontFamily = 'system-ui, -apple-system, sans-serif'
descriptionText.style.zIndex = '999'
descriptionText.style.pointerEvents = 'none'
descriptionText.style.lineHeight = '1.4'
descriptionText.style.textAlign = 'right'
descriptionText.style.letterSpacing = '0.3px'

// Create logo element
const logo = document.createElement('img')
logo.src = '/logo.png'
logo.style.position = 'fixed'
logo.style.top = '40px'
logo.style.left = '40px'
logo.style.zIndex = '999'
logo.style.height = '80px'
logo.style.width = 'auto'
logo.style.pointerEvents = 'none'

// Create navbar container
const navbar = document.createElement('nav')
navbar.style.position = 'fixed'
navbar.style.top = '50px'
navbar.style.left = '160px' // Positioned after the logo
navbar.style.zIndex = '999'
navbar.style.display = 'flex'
navbar.style.gap = '40px' // Space between menu items
navbar.style.alignItems = 'center'

// Create menu items
function createMenuItem(text: string) {
  const menuItem = document.createElement('div')
  menuItem.style.position = 'relative'
  menuItem.style.display = 'flex'
  menuItem.style.alignItems = 'center'
  menuItem.style.cursor = 'pointer'
  menuItem.style.color = '#ffffff'
  menuItem.style.fontSize = '16px'
  menuItem.style.fontWeight = '300'
  menuItem.style.fontFamily = 'system-ui, -apple-system, sans-serif'
  menuItem.style.letterSpacing = '0.5px'
  menuItem.style.transition = 'all 0.3s ease'

  // Create hover dot
  const dot = document.createElement('div')
  dot.style.width = '6px'
  dot.style.height = '6px'
  dot.style.borderRadius = '50%'
  dot.style.backgroundColor = '#ffffff'
  dot.style.marginRight = '12px'
  dot.style.opacity = '0'
  dot.style.transition = 'opacity 0.3s ease'
  dot.style.transform = 'scale(0)'
  dot.style.transition = 'opacity 0.3s ease, transform 0.3s ease'

  // Create text
  const textElement = document.createElement('span')
  textElement.textContent = text

  // Hover effects
  menuItem.addEventListener('mouseenter', () => {
    dot.style.opacity = '1'
    dot.style.transform = 'scale(1)'
    textElement.style.color = '#ffffff'
  })

  menuItem.addEventListener('mouseleave', () => {
    dot.style.opacity = '0'
    dot.style.transform = 'scale(0)'
  })

  menuItem.appendChild(dot)
  menuItem.appendChild(textElement)

  return menuItem
}

// Create menu items
const infoItem = createMenuItem('Info')
const aboutItem = createMenuItem('About')

// Add menu items to navbar
navbar.appendChild(infoItem)
navbar.appendChild(aboutItem)

// Create image carousel above hero text
const imageCarousel = document.createElement('div')
imageCarousel.style.position = 'fixed'
imageCarousel.style.top = '18%'
imageCarousel.style.left = '50%'
imageCarousel.style.transform = 'translateX(-50%)'
imageCarousel.style.zIndex = '999'
imageCarousel.style.width = '200px'
imageCarousel.style.height = '80px'
imageCarousel.style.display = 'flex'
imageCarousel.style.alignItems = 'center'
imageCarousel.style.justifyContent = 'center'

// Create image element
const carouselImage = document.createElement('img')
carouselImage.style.maxWidth = '100%'
carouselImage.style.maxHeight = '100%'
carouselImage.style.objectFit = 'contain'
carouselImage.style.filter = 'brightness(0) invert(1)' // Make images white
carouselImage.style.opacity = '0.8'
carouselImage.style.transition = 'opacity 0.5s ease'

// Image URLs array
const imageUrls = [
  'https://www.vos9x.com/footerLogos/alpine.svg',
  'https://www.vos9x.com/footerLogos/clear-street.svg',
  'https://www.vos9x.com/footerLogos/pillar.svg'
]

let currentImageIndex = 0

function switchImage() {
  // Fade out
  carouselImage.style.opacity = '0'

  setTimeout(() => {
    // Change image
    carouselImage.src = imageUrls[currentImageIndex]
    currentImageIndex = (currentImageIndex + 1) % imageUrls.length

    // Fade in
    carouselImage.style.opacity = '0.8'
  }, 250) // Half of transition time
}

// Initialize first image
carouselImage.src = imageUrls[0]
currentImageIndex = 1

// Switch images every 3 seconds
setInterval(switchImage, 3000)

// Append image to carousel
imageCarousel.appendChild(carouselImage)

// Create hero text container
const heroText = document.createElement('div')
heroText.style.position = 'fixed'
heroText.style.top = '40%'
heroText.style.left = '50%'
heroText.style.transform = 'translate(-50%, -50%)'
heroText.style.zIndex = '1000'
heroText.style.pointerEvents = 'none'
heroText.style.textAlign = 'center'

// First line
const heroLine1 = document.createElement('div')
heroLine1.textContent = 'We back the builders'
heroLine1.style.color = '#ffffff'
heroLine1.style.fontSize = '101px'
heroLine1.style.fontWeight = '400'
heroLine1.style.fontFamily = 'system-ui, -apple-system, sans-serif'
heroLine1.style.letterSpacing = '1px'
heroLine1.style.whiteSpace = 'nowrap'
heroLine1.style.lineHeight = '1.0'

// Second line
const heroLine2 = document.createElement('div')
heroLine2.textContent = 'of the future'
heroLine2.style.color = '#ffffff'
heroLine2.style.fontSize = '101px'
heroLine2.style.fontWeight = '400'
heroLine2.style.fontFamily = 'system-ui, -apple-system, sans-serif'
heroLine2.style.letterSpacing = '1px'
heroLine2.style.whiteSpace = 'nowrap'
heroLine2.style.lineHeight = '1.0'

// Append lines to container
heroText.appendChild(heroLine1)
heroText.appendChild(heroLine2)

// Create subtitle with rotating text
const subtitle = document.createElement('div')
subtitle.style.position = 'fixed'
subtitle.style.top = '55%'
subtitle.style.left = '50%'
subtitle.style.transform = 'translateX(-50%)'
subtitle.style.zIndex = '1000'
subtitle.style.pointerEvents = 'none'
subtitle.style.fontSize = '32px'
subtitle.style.fontWeight = '300'
subtitle.style.fontFamily = 'system-ui, -apple-system, sans-serif'
subtitle.style.color = '#87CEEB'
subtitle.style.whiteSpace = 'nowrap'
subtitle.style.letterSpacing = '0.5px'
subtitle.style.display = 'flex'
subtitle.style.alignItems = 'baseline'

// Create the text with special styling for "we look for"
const subtitleText = document.createElement('span')
subtitleText.innerHTML = 'Discover what&nbsp;'

const lookForText = document.createElement('span')
lookForText.innerHTML = 'we look for'
lookForText.style.position = 'relative'
lookForText.style.display = 'inline-block'

// Create shooting star stroke underline - thin to thick
const underline = document.createElement('div')
underline.style.position = 'absolute'
underline.style.bottom = '-8px'
underline.style.left = '0'
underline.style.width = '100%'
underline.style.height = '3px'
underline.style.background = 'linear-gradient(90deg, transparent 0%, #87CEEB 20%, #00BFFF 80%, #1E90FF 100%)'
underline.style.clipPath = 'polygon(0% 50%, 20% 0%, 100% 0%, 100% 100%, 20% 100%)'
underline.style.opacity = '0.9'
underline.style.filter = 'drop-shadow(0 0 3px #87CEEB)'

lookForText.appendChild(underline)

const inText = document.createElement('span')
inText.innerHTML = '&nbsp;in&nbsp;'

// Fixed-width container for rotating word to prevent shifting
const rotatingContainer = document.createElement('span')
rotatingContainer.style.display = 'inline-block'
rotatingContainer.style.width = '120px' // Fixed width to accommodate longest word
rotatingContainer.style.textAlign = 'left'

const rotatingWord = document.createElement('span')
const words = ['builders', 'founders', 'managers']
let currentWordIndex = 0
let isTyping = false

// Add typewriter animation styles
const typewriterStyle = document.createElement('style')
typewriterStyle.textContent = `
  @keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
  }
  .cursor {
    display: inline-block;
    background-color: #87CEEB;
    width: 2px;
    height: 1em;
    animation: blink 1s infinite;
    margin-left: 2px;
  }
`
document.head.appendChild(typewriterStyle)

// Create cursor
const cursor = document.createElement('span')
cursor.className = 'cursor'

function typeWriter(text: string, callback?: () => void) {
  if (isTyping) return
  isTyping = true

  // Clear current text
  rotatingWord.textContent = ''

  let i = 0
  const typeInterval = setInterval(() => {
    if (i < text.length) {
      rotatingWord.textContent += text.charAt(i)
      i++
    } else {
      clearInterval(typeInterval)
      isTyping = false
      if (callback) callback()
    }
  }, 50) // Typing speed
}

function updateRotatingWord() {
  if (!isTyping) {
    typeWriter(words[currentWordIndex], () => {
      setTimeout(() => {
        currentWordIndex = (currentWordIndex + 1) % words.length
        setTimeout(updateRotatingWord, 300) // Pause before next word
      }, 800) // Display time
    })
  }
}

// Initialize
updateRotatingWord()

// Assemble subtitle
rotatingContainer.appendChild(rotatingWord)
rotatingContainer.appendChild(cursor)

subtitle.appendChild(subtitleText)
subtitle.appendChild(lookForText)
subtitle.appendChild(inText)
subtitle.appendChild(rotatingContainer)

app.appendChild(gradientWrapper)
app.appendChild(hoverSpot)
app.appendChild(buttonWrapper)
app.appendChild(copyrightText)
app.appendChild(descriptionText)
app.appendChild(logo)
app.appendChild(navbar)
app.appendChild(imageCarousel)
app.appendChild(heroText)
app.appendChild(subtitle)

let time = 0


function animate() {
  requestAnimationFrame(animate)

  time += 0.01

  // Update neural network background
  updateNeuralBackground(time)

  // Auto-demo sequence - single transformation
  if (autoDemo && !demoCompleted && time > 0.2) {
    // Start transformation after 1 second (quick grid glimpse)
    demoCompleted = true
    isMountain = true
    targetProgress = 1
    // Keep text as 'Start'
  }

  transitionProgress += (targetProgress - transitionProgress) * 0.05

  // Smooth hover transition like element.txt
  const targetHover = isHovered ? 1.0 : 0.0
  hoverTransition += (targetHover - hoverTransition) * 0.1

  // Disable hover effect - show gradient normally
  shadowElement.style.mask = 'none'
  shadowElement.style.webkitMask = 'none'
  shadowElement.style.clipPath = 'none'
  shadowElement.style.opacity = '1'

  const positionAttribute = geometry.getAttribute('position')
  const positions = positionAttribute.array as Float32Array

  for (let i = 0; i < particleCount; i++) {
    const originalX = originalPositions[i * 3]
    const originalY = originalPositions[i * 3 + 1]
    const originalZ = originalPositions[i * 3 + 2]

    const mountainX = mountainPositions[i * 3]
    const mountainY = mountainPositions[i * 3 + 1]
    const mountainZ = mountainPositions[i * 3 + 2]

    // Add random pulsating movement to mountain particles with individual amplitudes (slower)
    const randomOffset = Math.sin(time * (0.5 + i * 0.0002)) * Math.cos(time * (0.3 + i * 0.0001)) * randomAmplitudes[i]
    const pulsatingMountainY = mountainY + randomOffset * transitionProgress

    // Simple interpolation with pulsating effect
    positions[i * 3] = originalX * (1 - transitionProgress) + mountainX * transitionProgress
    positions[i * 3 + 1] = originalY * (1 - transitionProgress) + pulsatingMountainY * transitionProgress
    positions[i * 3 + 2] = originalZ * (1 - transitionProgress) + mountainZ * transitionProgress
  }

  positionAttribute.needsUpdate = true

  // Adjust camera to more horizontal mountain view during transformation
  if (transitionProgress > 0.1) {
    const cameraY = 15 * transitionProgress // Lower camera height for more horizontal view
    const cameraZ = 50 + (30 * transitionProgress) // Move further back for mountain view
    camera.position.set(0, cameraY, cameraZ)
    camera.lookAt(0, 0, 0)
  } else {
    camera.position.set(0, 0, 50)
    camera.lookAt(0, 0, 0)
  }

  renderer.render(scene, camera)
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight
  camera.updateProjectionMatrix()
  renderer.setSize(window.innerWidth, window.innerHeight)

  // Update neural background resolution
  neuralMaterial.uniforms.iResolution.value.set(window.innerWidth, window.innerHeight)
})

animate()