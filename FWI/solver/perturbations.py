
import jax.numpy as jnp

def shepp_logan_transform(x1, y1, a1, b1, theta1, x2, y2, a2, b2, theta2, grid):
    ellipse_1 = (x1, y1, a1, b1, theta1, 0.8)
    ellipse_2 = (x2, y2, a2, b2, theta2, 0.8)
    return shepp_logan_2_inner(ellipse_1, ellipse_2, grid)


def shepp_logan_2_inner(ellipse1, ellipse2, grid_size=256):
    """
    Generate a Shepp-Logan phantom with two inner ellipses.

    Parameters:
    grid_size (int): The size of the square grid (grid_size x grid_size).
    ellipse1 (tuple): Parameters for the first ellipse (x_center, y_center, a, b, angle, intensity).
    ellipse2 (tuple): Parameters for the second ellipse (x_center, y_center, a, b, angle, intensity).

    Returns:
    jax.numpy.array: A grid with the phantom.
    """
    outer_ellipse_1 = (0.0, 0.0, 0.69, 0.92, 0.0, 1.0)
    outer_ellipse_2 = (0.0, -0.0184, 0.6624, 0.874, 0.0, 0.2)

    # Create a grid of x and y coordinates
    x = jnp.linspace(-1, 1, grid_size)
    y = jnp.linspace(-1, 1, grid_size)
    xx, yy = jnp.meshgrid(x, y)

    def add_ellipse(grid, ellipse):
        x_center, y_center, a, b, angle, intensity = ellipse

        # Transform coordinates to the rotated ellipse frame
        cos_angle = jnp.cos(jnp.pi*angle/180)
        sin_angle = jnp.sin(jnp.pi*angle/180)
        x_rot = cos_angle * (xx - x_center) + sin_angle * (yy - y_center)
        y_rot = -sin_angle * (xx - x_center) + cos_angle * (yy - y_center)

        # Compute the ellipse equation
        inside = ((x_rot / a)**2 + (y_rot / b)**2) <= 1
        return grid + intensity * inside

    # Initialize the grid
    grid = jnp.zeros((grid_size, grid_size))

    # Add two outer ellipses
    grid = add_ellipse(grid, outer_ellipse_1)
    grid = add_ellipse(grid, outer_ellipse_2)

    # Add the two ellipses to the grid
    grid = add_ellipse(grid, ellipse1)
    grid = add_ellipse(grid, ellipse2)

    return grid



def two_bumps(a,x,y):
    def perturbation(x,y):
        return    1.0*jnp.exp(-500*(jnp.square(x-0.1) + jnp.square(y-0.1)))\
                + 1.0*jnp.exp(-500*(jnp.square(x-0.15) + jnp.square(y+0.3)))
    return perturbation(x,y)


def three_bumps(x,y):
    return    1.0*jnp.exp(-500*(jnp.square(x+0.1) + jnp.square(y+0.2)))\
            + 1.0*jnp.exp(-500*(jnp.square(x-0.1) + jnp.square(y-0.1)))\
            + 1.0*jnp.exp(-500*(jnp.square(x-0.15) + jnp.square(y+0.3)))