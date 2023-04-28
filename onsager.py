import numpy as np
import ovito
from dataclasses import dataclass


@dataclass
class LinearRegression:

    slope: float
    intercept: float
    r_squared: float = None
    independent_var: np.ndarray = None
    dependent_var: np.ndarray = None


def linear_regression(x, y, callback: callable = None, r_squared_tol: float = 0.90):

    matrix = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(matrix, y, rcond=None)[0]
    r = np.corrcoef(x, y)[0, 1]
    
    regression = LinearRegression(slope, intercept, r ** 2, x, y)
    
    if callback is not None:
        callback(regression)
        
    if regression.r_squared < r_squared_tol:
    
        # try first 80 %
        end_index = int(0.8 * len(x))
        
        if end_index <= 10:
            raise ValueError
    
        x, y = x[:end_index], y[:end_index]
        return linear_regression(x, y, callback=callback, r_squared_tol=r_squared_tol)

    return regression


@dataclass
class ParticleAverageModifier:

    initial_attr: str
    final_attr: str
    type_dict: dict
    operation: callable = None

    def __call__(self, frame: int, data: ovito.data.DataCollection):

        types = data.particles['Particle Type'][...]
        per_particle_data = data.particles[self.initial_attr][...]

        for integer_label, element in self.type_dict.items():

            array = per_particle_data[types == integer_label]
            if self.operation is not None:
                array = self.operation(array)
            data.attributes[f'{self.final_attr}_{element}'] = np.mean(array)


class MSDModifier(ParticleAverageModifier):

    def __init__(self, type_dict):

        self.initial_attr = 'Displacement Magnitude'
        self.final_attr = 'msd'
        self.type_dict = type_dict
        self.operation = lambda x: x ** 2


class COMModifier(ParticleAverageModifier):

    def __init__(self, type_dict):

        self.initial_attr = 'Displacement'
        self.final_attr = 'com'
        self.type_dict = type_dict


@dataclass
class DiffusionCoefficientArray:

    array: np.ndarray
    type_dict: dict

    def __mul__(self, other):
        return DiffusionCoefficientArray(self.array * other, self.type_dict)

    __rmul__ = __mul__

    def __iter__(self):
        return zip(self.type_dict.values(), self.array)

    def __repr__(self):

        iterable = [f'{element}: {val}' for element, val in self]
        return ', '.join(iterable)

    def __getitem__(self, item):
        if item is Ellipsis:
            return np.array([val for _, val in self])
        elif item in self.type_dict.values():
            for index, element in enumerate(self.type_dict.values()):
                if item == element:
                    return self.array[index]
        elif item not in self.type_dict.values():
            raise ValueError(f'element {item} not found, must be an available element or Ellipses')


@dataclass
class OnsagerMatrix:

    matrix: np.ndarray
    type_dict: dict

    def __mul__(self, other):

        return OnsagerMatrix(other * self.matrix, self.type_dict)

    __rmul__ = __mul__

    def __getitem__(self, item):

        elements = list(self.type_dict.values())

        if item is Ellipsis:
            return self.matrix

        elif len(item) == 2 and item[0] in elements and item[1] in elements:

            integer_type_labels = []
            for x in item:
                idx = [i for i in self.type_dict if self.type_dict[i] == x]
                if len(idx) == 1:
                    integer_type_labels.append(idx[0])
                elif len(idx) == 0:
                    raise ValueError(f'{x} is not a valid element')
                elif len(idx) > 0:
                    raise ValueError(f'{x} is repeated {len(idx)} times, must not be repeated')

            indices = []
            integer_types = [val for val, _ in self.type_dict.items()]

            for label in integer_type_labels:
                length = np.max(self.matrix.shape)
                for index in range(length):
                    if label == integer_types[index]:
                        indices.append(index)

            return self.matrix[indices[0], indices[1]]

        elif len(item) != 2:
            raise ValueError(f'must provide 2 atom types when subscripting, not {len(item)}')

        elif item[0] not in elements or item[1] not in elements:
            raise ValueError(f'invalid atom types')

        else:
            raise ValueError(f'no support for subscripting unless argument is {Ellipsis}')

    def __repr__(self):

        val = ''

        types = [self.type_dict[type_] for type_ in self.type_dict]
        header = '\t'
        header += '\t'.join(types)
        header += '\n'
        val += header

        for type_, row in zip(types, self.matrix):
            printed_row = [f'{type_}']
            str_row = [str(x) for x in row]
            printed_row.extend(str_row)
            printed_row = '\t'.join(printed_row)
            printed_row += '\n'
            val += printed_row

        return val


@dataclass
class KineticProperties:

    pipeline: ovito.pipeline.Pipeline
    type_dict: dict
    time_step: float
    msd_attr_name: str = 'msd'
    com_attr_name: str = 'com'
    sample_size: int = 30
    regression_callback: callable = None

    def __post_init__(self):

        self.num_frames = self.pipeline.source.num_frames
        self.data = [self.pipeline.compute(frame) for frame in range(self.num_frames)]
        self.time = np.array([
            data.attributes['Timestep'] * self.time_step for data in self.data
        ])

    @property
    def diffusion_coefficients(self):

        msd_time_series = {element: np.zeros(self.pipeline.source.num_frames) for element in self.type_dict.values()}

        for frame, data in enumerate(self.data):
            for element in self.type_dict.values():
                msd_time_series[element][frame] = data.attributes[f'{self.msd_attr_name}_{element}']

        coefficients = np.zeros(len(msd_time_series.values()))

        for index, series in enumerate(msd_time_series.values()):
            regression = linear_regression(self.time, series, callback=self.regression_callback)
            coefficients[index] = regression.slope / 6.0

        return DiffusionCoefficientArray(coefficients, self.type_dict)

    @property
    def onsager_matrix(self):

        types = self.type_dict.values()
        num_types = len(types)
        matrix = np.zeros((num_types, num_types))

        mean_com_displacement_array = np.zeros((self.num_frames, len(types), 3))

        for frame, data in enumerate(self.data):
            for index, element in enumerate(types):
                mean_com_displacement_array[frame, index, :] = data.attributes[f'{self.com_attr_name}_{element}']

        rounded_time = int(self.sample_size * np.floor(len(self.time) / self.sample_size))
        time = self.time[:rounded_time // self.sample_size]

        for first_index, first_type in enumerate(types):
            for second_index, second_type in enumerate(types):

                first_displacement = mean_com_displacement_array[:, first_index, :]
                second_displacement = mean_com_displacement_array[:, second_index, :]

                # truncate after rounded time, makes splicing possible

                first_displacement = first_displacement[:rounded_time, :]
                second_displacement = second_displacement[:rounded_time, :]

                # splice displacements into N samples

                shape = (self.sample_size, -1, 3)
                first_reshaped = first_displacement.reshape(shape)
                second_reshaped = second_displacement.reshape(shape)
                iterable = enumerate(zip(first_reshaped, second_reshaped))

                for index, (first, second) in iterable:
                    first_reshaped[index] += -first[0]
                    second_reshaped[index] += -second[0]

                all_samples = np.zeros((self.sample_size, rounded_time // self.sample_size))
                iterable = enumerate(zip(first_reshaped, second_reshaped))

                for index, pair in iterable:
                    series = np.array([np.dot(*p) for p in zip(*pair)])
                    all_samples[index, :] = series

                mean_series = np.mean(all_samples, axis=0)
                regression = linear_regression(time, mean_series)
                matrix[first_index, second_index] = regression.slope / 6.0

        return OnsagerMatrix(matrix, self.type_dict)


def get_kinetic_properties(file_name, type_dict, time_step, regression_callback=None):

    pipeline = ovito.io.import_file(file_name)
    pipeline.modifiers.append(ovito.modifiers.CalculateDisplacementsModifier())
    for modifier in [MSDModifier, COMModifier]:
        pipeline.modifiers.append(modifier(type_dict))

    return KineticProperties(pipeline, type_dict, time_step, regression_callback=regression_callback)


def main():

    file_name = 'transV.dump'
    type_dict = {1: 'Fe', 2: 'Ni'}
    time_step = 0.002

    properties = get_kinetic_properties(file_name, type_dict, time_step, regression_callback=print)
    print('calculating diffusion coefficients')
    diffusion_coefficients = properties.diffusion_coefficients
    print('calculating onsager coefficients')
    onsager_matrix = properties.onsager_matrix

    # kinetic coefficients are in units of angstrom^2 per pico second, convert to cm^2/s

    conversion = (1e2 / 1e10) ** 2 / (1e0 / 1e12)
    diffusion_coefficients *= conversion
    onsager_matrix *= conversion

    print(diffusion_coefficients)
    print(onsager_matrix)


if __name__ == '__main__':

    main()
