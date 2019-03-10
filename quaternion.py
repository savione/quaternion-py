"""
quaternion.py
Cutter Coryell
2015-08-03

Defines two classes: Quaternion, which describes mathematical quaternions, and QList, which describes a list of quaternions with support for element-wise operations.
"""


import numpy as np

class Quaternion:
        
    def __init__(self, *args, **kwargs):
        """A Quaternion can be initialized with 0 to 4 arguments,
        representing the four components (w, x, y, z) in order
        (missing elements assumed to be 0). It can also be initialized
        with a tuple or list as an argument, in which case the first four
        elements will become the quaternion elements (w, x, y, z) in order
        (missing elements assumed to be 0). A quaternion can also be 
        initialized with keyword arguments "w", "x", "y", and/or "z".
        Mixing multiple methods of defining quaternion elements will
        result in unexpected behavior."""

        self.w = self.x = self.y = self.z = 0.

        try:
            self.w = args[0].w
            self.x = args[0].x
            self.y = args[0].y
            self.z = args[0].z
        except AttributeError:
            if len(args) == 4:
                self.w = args[0]
                self.x = args[1]
                self.y = args[2]
                self.z = args[3]
            elif len(args) == 3:
                self.w = 0.
                self.x = args[0]
                self.y = args[1]
                self.z = args[2]
            elif len(args) == 1:
                if len(args[0]) == 4:
                    self.w = args[0][0]
                    self.x = args[0][1]
                    self.y = args[0][2]
                    self.z = args[0][3]
                elif len(args[0]) == 3:
                    self.w = 0.
                    self.x = args[0][0]
                    self.y = args[0][1]
                    self.z = args[0][2]
            
            try:
                self.w = kwargs["w"]
            except KeyError:
                pass
            try:
                self.x = kwargs["x"]
            except KeyError:
                pass
            try:
                self.y = kwargs["y"]
            except KeyError:
                pass
            try:
                self.z = kwargs["z"]
            except KeyError:
                pass

        for element in (self.w, self.x, self.y, self.z):
            if not isinstance(element, int) and not isinstance(element, float):
                raise ValueError("All elements must be integers or floats.")

    def normsquared(self):
        return self.w**2 + self.x**2 + self.y**2 + self.z**2
    
    def __abs__(self):
        return self.normsquared()**0.5
    
    def __pos__(self):
        return self
    
    def __neg__(self):
        return Quaternion(-self.w, -self.x, -self.y, -self.z)
    
    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def inverse(self):
        return self.conjugate() / self.normsquared()    
    
    def real(self):
        return self.w
    
    def imaginary(self):
        return Quaternion(0, self.x, self.y, self.z)
    
    def scalar_part(self):
        return self.w

    def vector_part(self):
        return [self.x, self.y, self.z]

    def unit(self):
        return self / abs(self)
    
    def angle(self):
        """Returns the real angle `theta` such that self.unit() = cos(`theta`/2) + `u` * sin(`theta`/2)
        for some pure imaginary unit quaternion `u`."""
        return 2 * np.arccos(self.unit().real())
    
    def axis(self):
        """Returns the pure imaginary unit quaterion `u` such that self.unit() = cos(`theta`/2) + `u` * sin(`theta`/2)
        for some real angle `theta`."""
        return (self.unit() - self.unit().real()) / np.sin(np.arccos(self.unit().real()))

    def __eq__(self, other):
        try:
            w = other.w
            x = other.x
            y = other.y
            z = other.z
        except AttributeError:
            w = other
            x = y = z = 0
        return np.allclose([self.w, self.x, self.y, self.z], [w, x, y, z])
    
    def __ne__(self, other):
        return not self == other
    
    def __add__(self, other):
        try:
            w = other.w
            x = other.x
            y = other.y
            z = other.z
        except AttributeError:
            w = other
            x = y = z = 0
        return Quaternion(self.w + w, self.x + x, self.y + y, self.z + z)
    
    def __radd__(self, scalar):
        return self + scalar
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, scalar):
        return -self + scalar
    
    def __mul__(self, other):
        try:
            w = other.w
            x = other.x
            y = other.y
            z = other.z
        except AttributeError:
            w = other
            x = y = z = 0
        return Quaternion((self.w * w - self.x * x - self.y * y - self.z * z),
                          (self.w * x + self.x * w + self.y * z - self.z * y),
                          (self.w * y + self.y * w + self.z * x - self.x * z),
                          (self.w * z + self.z * w + self.x * y - self.y * x))

    def __rmul__(self, scalar):
        return self * scalar
    
    def __div__(self, other):
        try:
            w = other.w
            x = other.x
            y = other.y
            z = other.z
        except AttributeError:
            return (1. / other) * self
        return self * other.inverse()
    
    def __rdiv__(self, scalar):
        return scalar * self.inverse()
            
    def exp(self, *args, **kwargs):
        """Computes either e^self if there is no argument or `arg`^self if there is an argument `arg`."""
        if len(args) > 1:
            raise ValueError("only 0 or 1 arguments are allowed")
        try:
            base = args[0]
        except IndexError:
            base = np.e
        try:
            tolerange = kwargs["tolerance"]
        except KeyError:
            tolerance = 1e-40
        if abs(self.imaginary()) < tolerance:
            return base**self.w

        try:
            return base**self.w * (np.cos(np.log(base) * abs(self.imaginary())) 
                                   + (self.imaginary() / abs(self.imaginary()))
                                      * np.sin(np.log(base) * abs(self.imaginary())))
        except AttributeError:
            raise ValueError("argument must be a real number")
            
    def log(self):
        try:
            return np.log(abs(self)) + self.imaginary().unit() * np.arcsin(abs(self.imaginary()) / abs(self))
        except ZeroDivisionError:
            return np.log(abs(self))
    
    def __pow__(self, other):
        return (self.log() * other).exp()
    
    def __repr__(self):
        basis_strings = ('', 'i', 'j', 'k')
        representation = ""
        for value, basis in zip((self.w, self.x, self.y, self.z), basis_strings):
            if representation:
                if value < 0:
                        prefix = " - "
                else:
                    prefix = " + "
            else:
                if value < 0:
                    prefix = "-"
                else:
                    prefix = ""
            if value:
                if abs(value) == 1 and basis != '':
                    representation += (prefix + "{}".format(basis))
                else:
                    representation += (prefix + "{}{}".format(abs(value), basis))
        if not representation:
            representation = "0"
        return representation
    
    @classmethod 
    def rotation(class_, angle, x, y, z):
        """Generates rotation quaternion for rotating by `angle` along the axis `(x, y, z)` points along."""
        u = class_(0, x, y, z)
        u /= abs(u)
        return (0.5 * angle * u).exp()
    
Q1 = Quaternion(1, 0, 0, 0) # 1 as a quaternion
i = Quaternion(0, 1, 0, 0)
j = Quaternion(0, 0, 1, 0)
k = Quaternion(0, 0, 0, 1)

class QList:
    """
    Extension of the built-in list type for quaternions, allowing element-wise quaternion operations.
    """

    def __init__(self, sequence):
        self.quaternions = [Quaternion(obj) for obj in sequence]

    def __len__(self):
        return len(self.quaternions)

    def __getitem__(self, key):
        return self.quaternions[key]

    def __setitem__(self, key, value):
        self.quaternions[key] = value

    def __delitem__(self, key):
        del self.quaternions[key]

    def __iter__(self):
        return iter(self.quaternions)

    def normsquared(self):
        return np.array([q.normsquared() for q in self])
    
    def __abs__(self):
        return np.array([abs(q) for q in self])
    
    def __pos__(self):
        return QList([+q for q in self])
    
    def __neg__(self):
        return QList([-q for q in self])
    
    def conjugate(self):
        return QList([q.conjugate() for q in self])
    
    def inverse(self):
        return QList([q.inverse() for q in self])    
    
    def real(self):
        return np.array([q.real() for q in self])
    
    def imaginary(self):
        return QList([q.imaginary() for q in self])
    
    def scalar_part(self):
        return np.array([q.scalar_part() for q in self])
    
    def vector_part(self):
        return np.array([q.vector_part() for q in self])

    def unit(self):
        return QList([q.unit() for q in self])
    
    def angle(self):
        """Returns the real angle `theta` such that self.unit() = cos(`theta`/2) + `u` * sin(`theta`/2)
        for some pure imaginary unit quaternion `u`."""
        return np.array([q.angle() for q in self])
    
    def axis(self):
        """Returns the pure imaginary unit quaterion `u` such that self.unit() = cos(`theta`/2) + `u` * sin(`theta`/2)
        for some real angle `theta`."""
        return QList([q.axis() for q in self])

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for this_quaternion, that_quaternion in zip(self, other):
            if this_quaternion != that_quaternion:
                return False
        return True    

    def __ne__(self, other):
        return not self == other
    
    def __add__(self, other):
        try:
            if len(self) != len(other):
                raise ValueError("the two QLists must be of equal length")
        except TypeError:
            return QList([q + other for q in self])
        else:
            return QList([q1 + q2 for q1, q2 in zip(self, other)])
    
    def __radd__(self, quaternion):
        return self + quaternion
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, quaternion):
        return -self + quaternion
    
    def __mul__(self, other):
        try:
            if len(self) != len(other):
                raise ValueError("the two QLists must be of equal length")
        except TypeError:
            return QList([q * other for q in self])
        else:
            return QList([q1 * q2 for q1, q2 in zip(self, other)])
    
    def __rmul__(self, quaternion):
        return QList([quaternion * q for q in self])
    
    def __div__(self, other):
        try:
            if len(self) != len(other):
                raise ValueError("the two QLists must be of equal length")
        except TypeError:
            return QList([q / other for q in self])
        else:
            return QList([q1 / q2 for q1, q2 in zip(self, other)])
    
    def __rdiv__(self, quaternion):
        return QList([quaternion / q for q in self])
            
    def exp(self, *args, **kwargs):
        try:
            other = args[0]
        except IndexError:
            return QList([q.exp(**kwargs) for q in self])
        else:
            try:
                if len(self) != len(args[0]):
                    raise ValueError("the two QLists must be of equal length")
            except TypeError:
                return QList([q.exp(other, **kwargs) for q in self])
            else:
                return QList([q1.exp(q2, **kwargs) for q1, q2
                                                   in zip(self, other)])
            
    def log(self):
        return QList([q.log() for q in self])
    
    def __pow__(self, other):
        return (self.log() * other).exp()
    
    def __repr__(self):
        return str(self.quaternions)
    
# (c) Cutter Coryell 2015. All rights reserved.