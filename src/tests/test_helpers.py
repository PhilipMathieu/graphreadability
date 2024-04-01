import numpy as np
import networkx as nx
import src.utils.helpers as helpers
import unittest


class TestHelpers(unittest.TestCase):

    def setUp(self):
        """
        The graph should look like this (roughly):

        2------1       5
        | \\  /
        |  \\/
        |  /\\
        | /  \\
        3------4

        With the crossing edges intersecting at the origin.
        """
        self.graph = nx.Graph()
        self.graph.add_nodes_from(
            [
                (1, {"x": 1, "y": 1}),
                (2, {"x": -1, "y": 1}),
                (3, {"x": -1, "y": -1}),
                (4, {"x": 1, "y": -1}),
                (5, {"x": 2, "y": 1}),
            ]
        )
        self.graph.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 2), (1, 3)])

    def test_is_positive(self):
        self.assertTrue(helpers._is_positive(1))
        self.assertTrue(helpers._is_positive(0.0001))
        self.assertFalse(helpers._is_positive(0))
        self.assertFalse(helpers._is_positive(-0.0001))
        array = np.array([1, 0.0001, 0, -0.0001])
        np.testing.assert_array_equal(
            helpers._is_positive(array), [True, True, False, False]
        )

    def test_divide_or_zero(self):
        a, b = 1, 2
        self.assertEqual(helpers.divide_or_zero(a, b), a / b)
        self.assertEqual(helpers.divide_or_zero(a, 0), 0)
        self.assertEqual(helpers.divide_or_zero(0, b), 0)
        self.assertEqual(helpers.divide_or_zero(0, 0), 0)

    def test_angle_between_vectors(self):
        a = np.array([1, 0])
        b = np.array([0, 1])
        np.testing.assert_almost_equal(
            helpers.calculate_angle_between_vectors(a, b), 90
        )
        a = np.array([1, 0])
        b = np.array([1, 0])
        np.testing.assert_almost_equal(helpers.calculate_angle_between_vectors(a, b), 0)
        a = np.array([1, 0])
        b = np.array([-1, 0])
        np.testing.assert_almost_equal(
            helpers.calculate_angle_between_vectors(a, b), 180
        )
        a = np.array([1, 0])
        b = np.array([1, 1])
        np.testing.assert_almost_equal(
            helpers.calculate_angle_between_vectors(a, b), 45
        )

    def test_in_circle(self):
        self.assertTrue(helpers._in_circle(0, 0, 0, 0, 1))
        self.assertFalse(helpers._in_circle(1, 1, 0, 0, 1))
        self.assertTrue(helpers._in_circle(0, 1, 0, 0, 1))
        self.assertTrue(helpers._in_circle(1, 0, 0, 0, 1))
        self.assertFalse(helpers._in_circle(2, 2, 0, 0, 1))

    def test_are_collinear_points(self):
        a = np.array([0, 0])
        b = np.array([1, 1])
        c = np.array([2, 2])
        self.assertTrue(helpers._are_collinear_points(a, b, c))
        a = np.array([0, 0])
        b = np.array([1, 1])
        c = np.array([2, 3])
        self.assertFalse(helpers._are_collinear_points(a, b, c))

    def test_rel_point_line_dist(self):
        axis = np.array([[0, 0], [1, 1]])
        x = 0
        y = 0
        self.assertEqual(helpers._rel_point_line_dist(axis, x, y), 0)
        x = 1
        y = 1
        self.assertEqual(helpers._rel_point_line_dist(axis, x, y), 0)
        x = 0.5
        y = 0.5
        self.assertEqual(helpers._rel_point_line_dist(axis, x, y), 0)
        x = 2
        y = 2
        self.assertEqual(helpers._rel_point_line_dist(axis, x, y), 0)
        x = 1
        y = 0
        self.assertAlmostEqual(helpers._rel_point_line_dist(axis, x, y), np.sqrt(2) / 2)

    def test_euclidean_distance(self):
        a = np.array([0, 0])
        b = np.array([3, 4])
        self.assertEqual(helpers._euclidean_distance(a, b), 5)

    def test_same_distance(self):
        a = 1
        b = 1.5
        self.assertEqual(helpers._same_distance(a, b), True)
        a = -1
        b = 1
        self.assertEqual(helpers._same_distance(a, b), True)
        a = 1
        b = 2
        self.assertEqual(helpers._same_distance(a, b), False)

    def test_bounding_box_nd(self):
        points = np.array([[0, 0], [1, 1], [2, 2]])
        self.assertTrue(
            np.array_equal(helpers._bounding_box_nd(points), np.array([[0, 0], [2, 2]]))
        )

    def test_midpoint_nd(self):
        a = np.array([0, 0])
        b = np.array([2, 2])
        self.assertTrue(np.array_equal(helpers._midpoint_nd(a, b), np.array([1, 1])))

    def test_circles_intersect_nd(self):
        c1 = np.array([0, 0])
        c2 = np.array([2, 2])
        r1 = np.sqrt(2)
        r2 = np.sqrt(2)
        self.assertTrue(helpers._circles_intersect_nd(c1, c2, r1, r2))
        c1 = np.array([0, 0, 0])
        c2 = np.array([2, 2, 2])
        self.assertFalse(helpers._circles_intersect_nd(c1, c2, r1, r2))

    def test_circles_intersect(self):
        x1 = 0
        y1 = 0
        x2 = 2
        y2 = 2
        r1 = np.sqrt(2)
        r2 = np.sqrt(2)
        self.assertTrue(helpers._circles_intersect(x1, y1, x2, y2, r1, r2))
        x2 = 3
        y2 = 3
        self.assertFalse(helpers._circles_intersect(x1, y1, x2, y2, r1, r2))

    def test_in_rectangle(self):
        point = np.array([1, 1])
        minima = np.array([0, 0])
        maxima = np.array([2, 2])
        self.assertEqual(helpers._in_rectangle(point, minima, maxima), True)
        point = np.array([3, 3])
        minima = np.array([0, 0])
        maxima = np.array([2, 2])
        self.assertEqual(helpers._in_rectangle(point, minima, maxima), False)

    def test_is_point_inside_square(self):
        x = 1
        y = 1
        x1 = 0
        y1 = 0
        x2 = 2
        y2 = 2
        self.assertEqual(helpers._is_point_inside_square(x, y, x1, y1, x2, y2), True)
        x = 3
        y = 3
        x1 = 0
        y1 = 0
        x2 = 2
        y2 = 2
        self.assertEqual(helpers._is_point_inside_square(x, y, x1, y1, x2, y2), False)

    def test_on_opposite_sides(self):
        a = np.array([0, 1])
        b = np.array([1, 0])
        line = np.array([[0, 0], [1, 1]])
        self.assertTrue(helpers._on_opposite_sides(a, b, line))
        a = np.array([0, 0])
        self.assertFalse(helpers._on_opposite_sides(a, b, line))
        a = np.array([1, 0.5])
        self.assertFalse(helpers._on_opposite_sides(a, b, line))
        a = np.array([0.5, 0.5])
        b = np.array([0.5, 0.5])
        self.assertFalse(helpers._on_opposite_sides(a, b, line))

    def test_bounding_box(self):
        line_a = np.array([[0, 0], [1, 1]])
        line_b = np.array([[0, 1], [1, 0]])
        self.assertTrue(helpers._bounding_box(line_a, line_b))
        line_a = np.array([[2, 2], [1, 1]])
        self.assertTrue(helpers._bounding_box(line_a, line_b))
        line_a = np.array([[2, 2], [3, 3]])
        self.assertFalse(helpers._bounding_box(line_a, line_b))
        line_a = np.array([[-1, -1], [0.1, 0.1]])
        self.assertTrue(helpers._bounding_box(line_a, line_b))

    def test_find_k_nearest_points(self):
        p = np.array([0, 0])
        points = np.array([[-0.5, -0.5], [0, 0], [1, 1], [2, 2]])
        k = 1
        self.assertTrue(
            np.array_equal(
                helpers._find_k_nearest_points(p, points, k), np.array([0, 0])
            )
        )
        k = 2
        self.assertTrue(
            np.array_equal(
                helpers._find_k_nearest_points(p, points, k),
                np.array([[0, 0], [-0.5, -0.5]]),
            )
        )
        k = 3
        self.assertTrue(
            np.array_equal(
                helpers._find_k_nearest_points(p, points, k),
                np.array([[0, 0], [-0.5, -0.5], [1, 1]]),
            )
        )
        k = 4
        self.assertTrue(
            np.array_equal(
                helpers._find_k_nearest_points(p, points, k),
                np.array([[0, 0], [-0.5, -0.5], [1, 1], [2, 2]]),
            )
        )
        k = 0  # Should throw an error
        with self.assertRaises(ValueError):
            helpers._find_k_nearest_points(p, points, k)
        k = 5
        with self.assertRaises(IndexError):
            helpers._find_k_nearest_points(p, points, k)

    def test_lines_intersect(self):
        line_a = np.array([[0, 0], [1, 1]])
        line_b = np.array([[0, 1], [1, 0]])
        self.assertTrue(helpers._lines_intersect(line_a, line_b))
        line_b = np.array([[0, 2], [1, 1]])
        self.assertFalse(helpers._lines_intersect(line_a, line_b))
        line_b = line_a
        self.assertTrue(helpers._lines_intersect(line_a, line_b))

    def test_intersect(self):
        line_a = np.array([[0, 0], [1, 1]])
        line_b = np.array([[0, 1], [1, 0]])
        self.assertTrue(helpers._intersect(line_a, line_b))
        line_b = np.array([[0, 2], [1, 1]])
        self.assertFalse(helpers._intersect(line_a, line_b))
        line_b = line_a
        self.assertTrue(helpers._intersect(line_a, line_b))

    def test_compute_intersection(self):
        p1 = np.array([0, 0])
        q1 = np.array([1, 1])
        p2 = np.array([0, 1])
        q2 = np.array([1, 0])
        self.assertTrue(
            np.array_equal(
                helpers._compute_intersection(p1, q1, p2, q2), np.array([0.5, 0.5])
            )
        )
        p1 = np.array([0, 0])
        q1 = np.array([1, 1])
        p2 = np.array([0, 2])
        q2 = np.array([1, 1])
        self.assertTrue(
            np.array_equal(
                helpers._compute_intersection(p1, q1, p2, q2), np.array([1, 1])
            )
        )

    def test_same_position(self):
        n1 = 1
        n2 = 1
        self.assertTrue(helpers._same_position(n1, n2, self.G))
        n2 = 2
        self.assertFalse(helpers._same_position(n1, n2, self.G))

    def test_are_collinear(self):
        n1 = 1
        n2 = 2
        n3 = 3
        self.assertFalse(helpers._are_collinear(n1, n2, n3, self.G))
        n3 = 5
        self.assertTrue(helpers._are_collinear(n1, n2, n3, self.G))
