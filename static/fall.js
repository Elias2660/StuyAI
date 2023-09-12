const { Engine, Render, Runner, Bodies, Composite } = Matter;

const engine = Engine.create();

const render = Render.create({
  element: document.getElementById('waving'),
  engine: engine,
  options: {
    width: 400,
    height: 250,
    background: 'transparent',
    wireframes: false // To show filled shapes instead of wireframes
  }
});

// Create walls (sides and bottom)
const wallOptions = { isStatic: true, render: { fillStyle: 'black' } };
const leftWall = Bodies.rectangle(0, 125, 1, 250, wallOptions);
const rightWall = Bodies.rectangle(400, 125, 1, 250, wallOptions);
const bottomWall = Bodies.rectangle(200, 250, 400, 1, wallOptions);

const groundWidth = 200; // Adjust the width of the ground (box)
const groundHeight = 0; // Adjust the height of the ground (box)
const ground = Bodies.rectangle(200, 247.5, groundWidth, groundHeight, {
  isStatic: true,
  render: { fillStyle: 'black' }
});

Composite.add(engine.world, [leftWall, rightWall, bottomWall, ground]);

const runner = Runner.create();

Runner.run(runner, engine);
Render.run(render);

// Function to create shapes and add them to the world
function createShape(x, y) {
  const sides = Math.floor(Math.random() * 4) + 3; // Random number of sides (3 to 6)
  const radius = 40; // Increase the radius to make the shapes larger
  const shape = Bodies.polygon(x, y, sides, radius, {
    frictionAir: 0.05,
    restitution: 0.6,
    render: {
      fillStyle: getRandomBlueColor()
    }
  });

  // Set random initial velocity
  const angle = Math.random() * Math.PI * 2;
  const velocity = 10; // Adjust the speed of the shapes
  shape.velocity.x = velocity * Math.cos(angle);
  shape.velocity.y = velocity * Math.sin(angle);

  Composite.add(engine.world, shape);
}

// Function to generate a random blue color
function getRandomBlueColor() {
  const randomShade = Math.floor(Math.random() * 100);
  return `rgb(30, 30, ${100 + randomShade})`;
}

// Create shapes every 1 second
setInterval(() => createShape(Math.random() * 400, -50), 1000);
