import unittest
from math import sqrt
from vector import Tuple, Matrix, point, vector, Canvas, Color

#def matrix_to_list(precision):

class DadsTest(unittest.TestCase):

	def assertApproxMatrixEquals(self, m1, m2):
		self.assertEqual()

	def test_vec_add(self):

		t1 = Tuple(3, -2, 5, 1)
		t2 = Tuple(-2, 3, 1, 0)
		self.assertEqual(t1.add(t2), Tuple(1, 1, 6, 1))

	def test_vec_sub(self):
		t1 = Tuple(5, 4, 3, 2)
		t2 = Tuple(10, 0, 3, 1)
		self.assertEqual(t1.sub(t2), Tuple(-5, 4, 0, 1))

	def test_vec_neg(self):
		t1 = Tuple(1, -2, 3, -4)
		self.assertEqual(t1.negate(), Tuple(-1, 2, -3, 4))

	def test_vec_mul(self):
		t1 = Tuple(1, 2, 3, 4)
		self.assertEqual(t1.mul(3), Tuple(3, 6, 9, 12))

	def test_vec_div(self):
		t1 = Tuple(2, 3, 4, 5)
		self.assertEqual(t1.div(2), Tuple(1, 1.5, 2, 2.5))

	def test_magnitude1(self):
		v = vector(1, 0, 0)
		self.assertEqual(v.magnitude(), 1)

	def test_magnitude2(self):
		v = vector(1, 2, 3)
		self.assertEqual(v.magnitude(), sqrt(14))

	def test_magnitude3(self):
		v = vector(-1, -2, -3)
		self.assertEqual(v.magnitude(), sqrt(14))

	def test_normalize1(self):
		v = vector(4, 0, 0);
		self.assertEqual(v.normalize(), vector(1, 0, 0))

	def test_normalize2(self):
		v = vector(1, 2, 3);
		self.assertEqual(v.normalize(), vector(1/sqrt(14), 2/sqrt(14), 3/sqrt(14)))

	def test_dot1(self):
		self.assertEqual(vector(1, 2,3).dot(vector(2, 3, 4)), 20)

	def test_cross1(self):
		v1 = vector(1, 2, 3)
		v2 = vector(2, 3, 4)
		self.assertEqual(v1.cross(v2), vector(-1, 2, -1))
		self.assertEqual(v2.cross(v1), vector(1, -2, 1))
	
	def test_canvas(self):
		c = Canvas(5, 3)
		c.write_pixel(0, 0, Color(1.5, 0, 0))
		c.write_pixel(2, 1, Color(0, 0.5, 0))
		c.write_pixel(4, 2, Color(-0.5, 0, 1))
		c.save("test1.ppm")


	def test_matrix_mul(self):
		m1 = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6], [5, 4, 3, 2]])
		m2 = Matrix([[-2, 1, 2, 3], [3, 2, 1, -1], [4, 3, 6, 5], [1, 2, 7, 8]])
		answer = Matrix([[20, 22, 50, 48], [44, 54, 114, 108], [40, 58, 110, 102], [16, 26, 46, 42]])
		self.assertEqual(answer, m1.mul(m2))

	def test_matrix_transpose(self):
		m1 = Matrix([[0, 9, 3, 0], [9, 8, 0, 8], [1, 8, 5, 3], [0, 0, 5, 8]])
		answer = Matrix([[0, 9, 1, 0], [9, 8, 8, 0], [3, 0, 5, 5], [0, 8, 3, 8]])
		self.assertEqual(answer, m1.transpose())

	def test_submatrix1(self):
		m1 = Matrix([[1, 5, 0], [-3, 2, 7], [0, 6, -3]])
		answer = Matrix([[-3, 2], [0, 6]])
		self.assertEqual(answer, m1.submatrix(0, 2))

	def test_submatrix2(self):
		m1 = Matrix([[-6, 1, 1, 6], [-8, 5, 8, 6], [-1, 0, 8, 2], [-7, 1, -1, 1]]);
		answer = Matrix([[-6, 1, 6], [-8, 8, 6], [-7, -1, 1]])
		self.assertEqual(answer, m1.submatrix(2, 1))

	def test_minor1(self):
		m1 = Matrix([[3, 5, 0], [2, -1, -7], [6, -1, 5]])
		self.assertEqual(m1.minor(1, 0), 25)

	def test_determinant1(self):
		m = Matrix([[1, 5], [-3, 2]])
		self.assertEqual(17, m.determinant())

	def test_determinant2(self):
		m = Matrix([[1, 2, 6], [-5, 8, -4], [2, 6, 4]])
		self.assertEqual(-196, m.determinant())

	def test_determinant3(self):
		m = Matrix([[-2, -8, 3, 5], [-3, 1, 7, 3], [1, 2, -9, 6], [-6, 7, 7, -9]])
		self.assertEqual(-4071, m.determinant())

	def test_invertable1(self):
		m = Matrix([[6, 4, 4, 4], [5, 5, 7, 6], [4, -9, 3, -7], [9, 1, 7, -6]])
		self.assertEqual(-2120, m.determinant())
		self.assertTrue(m.is_invertable())

	def test_invertable2(self):
		m = Matrix([[-4, 2, -2, -3], [9, 6, 2, 6], [0, -5, 1, -5], [0, 0, 0, 0]])
		self.assertEqual(0, m.determinant())
		self.assertFalse(m.is_invertable())

	def test_inverse1(self):
		m = Matrix([[-5, 2, 6, -8], [1, -5, 1, 8], [7, 7, -6, -7], [1, -3, 7, 4]])
		self.assertEqual(532, m.determinant())
		m_inverse = m.invert()

		answer = Matrix([
			[0.21805, 0.45113, 0.24060, -0.04511], 
			[-0.80827, -1.45677, -0.44361, 0.52068],
			[-0.07895, -0.22368, -0.05263, 0.19737],
			[-0.52256, -0.81391, -0.30075, 0.30639]])

	#		self.assetApproxEqual(m, answer)

if __name__ == '__main__':
	unittest.main()

