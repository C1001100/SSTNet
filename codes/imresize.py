import numpy as np
from scipy.ndimage import filters, measurements, interpolation
from math import pi


def imresize(im, scale_factor=None, output_shape=None, kernel=None, antialiasing=True, kernel_shift_flag=False):
    # First standardize values and fill missing arguments (if needed) by deriving scale from output shape or vice versa
    # 首先通过从输出形状派生比例来标准化值并填充缺失的参数（如果需要），反之亦然
    scale_factor, output_shape = fix_scale_and_size(im.shape, output_shape, scale_factor)

    # For a given numeric kernel case, just do convolution and sub-sampling (downscaling only)
    # 对于给定的数字核情况，只需进行卷积和子采样（仅限缩小）
    if type(kernel) == np.ndarray and scale_factor[0] <= 1:
        return numeric_kernel(im, kernel, scale_factor, output_shape, kernel_shift_flag)

    # Choose interpolation method, each method has the matching kernel size
    # 选择插值法，每种方法都有匹配的内核大小
    method, kernel_width = {
        "cubic": (cubic, 4.0),
        "lanczos2": (lanczos2, 4.0),
        "lanczos3": (lanczos3, 6.0),
        "box": (box, 1.0),
        "linear": (linear, 2.0),
        None: (cubic, 4.0)  # set default interpolation method as cubic
    }.get(kernel)

    # Antialiasing is only used when downscaling
    # 抗锯齿仅在缩小时使用
    antialiasing *= (scale_factor[0] < 1)

    # Sort indices of dimensions according to scale of each dimension. since we are going dim by dim this is efficient
    # 根据每个维度的比例对维度索引进行排序。由于我们逐维度，这是有效的
    sorted_dims = np.argsort(np.array(scale_factor)).tolist()

    # Iterate over dimensions to calculate local weights for resizing and resize each time in one direction
    # 遍历维度以计算局部权重，以便每次在一个方向上调整大小和大小
    out_im = np.copy(im)
    for dim in sorted_dims:
        # No point doing calculations for scale-factor 1. nothing will happen anyway
        # 对比例因子 1 进行计算毫无意义。反正什么都不会发生
        if scale_factor[dim] == 1.0:
            continue

        # for each coordinate (along 1 dim), calculate which coordinates in the input image affect its result and the
        # weights that multiply the values there to get its result.
        # 对于每个坐标（沿 1 DIM 方向），计算输入图像中的哪些坐标会影响其结果，以及将值相乘以获得结果的权重。
        weights, field_of_view = contributions(im.shape[dim], output_shape[dim], scale_factor[dim],
                                               method, kernel_width, antialiasing)

        # Use the affecting position values and the set of weights to calculate the result of resizing along this 1 dim
        # 使用影响位置值和权重集来计算沿此 1 暗淡调整大小的结果
        out_im = resize_along_dim(out_im, dim, weights, field_of_view)

    return out_im


def fix_scale_and_size(input_shape, output_shape, scale_factor):
    # First fixing the scale-factor (if given) to be standardized the function expects (a list of scale factors in the
    # same size as the number of input dimensions)
    # 首先确定要标准化的比例因子（如果给定）函数期望（与输入维度数量大小相同的比例因子列表）
    if scale_factor is not None:
        # By default, if scale-factor is a scalar we assume 2d resizing and duplicate it.
        # 默认情况下，如果比例因子是标量，我们假设 2D 调整大小并复制它。
        if np.isscalar(scale_factor):
            scale_factor = [scale_factor, scale_factor]

        # We extend the size of scale-factor list to the size of the input by assigning 1 to all the unspecified scales
        # 我们通过为所有未指定的比例分配 1 来将比例因子列表的大小扩展到输入的大小
        scale_factor = list(scale_factor)
        scale_factor.extend([1] * (len(input_shape) - len(scale_factor)))

    # Fixing output-shape (if given): extending it to the size of the input-shape, by assigning the original input-size
    # to all the unspecified dimensions
    # 固定输出形状（如果给定）：通过将原始输入大小分配给所有未指定的维度，将其扩展到输入形状的大小
    if output_shape is not None:
        output_shape = list(np.uint(np.array(output_shape))) + list(input_shape[len(output_shape):])

    # Dealing with the case of non-give scale-factor, calculating according to output-shape. note that this is
    # sub-optimal, because there can be different scales to the same output-shape.
    # 处理非给定比例因子的情况，根据输出形状计算。请注意，这是次优的，因为同一输出形状可以有不同的比例。
    if scale_factor is None:
        scale_factor = 1.0 * np.array(output_shape) / np.array(input_shape)

    # Dealing with missing output-shape. calculating according to scale-factor
    # 处理缺少的输出形状。根据比例因子计算
    if output_shape is None:
        output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))

    return scale_factor, output_shape


def contributions(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    # This function calculates a set of 'filters' and a set of field_of_view that will later on be applied
    # such that each position from the field_of_view will be multiplied with a matching filter from the
    # 'weights' based on the interpolation method and the distance of the sub-pixel location from the pixel centers
    # around it. This is only done for one dimension of the image.
    # 此函数计算一组“过滤器”和一组稍后将应用的field_of_view
    # 使得field_of_view中的每个位置都将乘以来自
    # 基于插值方法和子像素位置与像素中心距离的“权重”
    # 围绕它。这仅适用于图像的一个维度。

    # When anti-aliasing is activated (default and only for downscaling) the receptive field is stretched to size of
    # 1/sf. this means filtering is more 'low-pass filter'.
    # 激活抗锯齿（默认且仅用于缩小）时，感受野被拉伸到大小
    # 1/SF. 这意味着滤波更像是“低通滤波器”。
    fixed_kernel = (lambda arg: scale * kernel(scale * arg)) if antialiasing else kernel
    kernel_width *= 1.0 / scale if antialiasing else 1.0

    # These are the coordinates of the output image
    # 这些是输出图像的坐标
    out_coordinates = np.arange(1, out_length+1)

    # These are the matching positions of the output-coordinates on the input image coordinates.
    # Best explained by example: say we have 4 horizontal pixels for HR and we downscale by SF=2 and get 2 pixels:
    # [1,2,3,4] -> [1,2]. Remember each pixel number is the middle of the pixel.
    # The scaling is done between the distances and not pixel numbers (the right boundary of pixel 4 is transformed to
    # the right boundary of pixel 2. pixel 1 in the small image matches the boundary between pixels 1 and 2 in the big
    # one and not to pixel 2. This means the position is not just multiplication of the old pos by scale-factor).
    # So if we measure distance from the left border, middle of pixel 1 is at distance d=0.5, border between 1 and 2 is
    # at d=1, and so on (d = p - 0.5).  we calculate (d_new = d_old / sf) which means:
    # (p_new-0.5 = (p_old-0.5) / sf)     ->          p_new = p_old/sf + 0.5 * (1-1/sf)
    # 这些是输入图像坐标上输出坐标的匹配位置。
    # 最好用例子来解释：假设我们有 4 个水平像素用于 HR，我们缩小 SF=2 并得到 2 个像素：
    # [1，2，3，4] -> [1，2]。请记住，每个像素数都是像素的中间。
    # 缩放是在距离之间完成的，而不是在像素数之间完成的（像素 4 的右边界转换为
    # 像素 2 的右边界。小图像中的像素 1 与大图像中的像素 1 和 2 之间的边界匹配
    # 一个而不是像素 2。这意味着位置不仅仅是将旧的pos乘以比例因子）。
    # 因此，如果我们测量距左边框的距离，像素 1 的中间距离为 d=0.5，1 和 2 之间的边界为
    # 在 d=1 时，依此类推 （d = p - 0.5）。 我们计算 （d_new = d_old / SF），这意味着：
    # （p_new-0.5 = （p_old-0.5） / 平方英尺） -> p_new = p_old/平方英尺 + 0.5 * （1-1/平方英尺）
    match_coordinates = 1.0 * out_coordinates / scale + 0.5 * (1 - 1.0 / scale)

    # This is the left boundary to start multiplying the filter from, it depends on the size of the filter
    # 这是开始乘以过滤器的左边界，这取决于过滤器的大小
    left_boundary = np.floor(match_coordinates - kernel_width / 2)

    # Kernel width needs to be enlarged because when covering has sub-pixel borders, it must 'see' the pixel centers
    # of the pixels it only covered a part from. So we add one pixel at each side to consider (weights can zeroize them)
    # 内核宽度需要放大，因为当覆盖具有子像素边框时，它必须“看到”像素中心
    # 的像素，它只覆盖了一部分。所以我们在每边添加一个像素来考虑（权重可以将它们归零）
    expanded_kernel_width = np.ceil(kernel_width) + 2

    # Determine a set of field_of_view for each each output position, these are the pixels in the input image
    # that the pixel in the output image 'sees'. We get a matrix whos horizontal dim is the output pixels (big) and the
    # vertical dim is the pixels it 'sees' (kernel_size + 2)
    # 为每个输出位置确定一组field_of_view，这些是输入图像中的像素
    # 输出图像中的像素“看到”的。我们得到一个水平调暗的矩阵是输出像素（大）和
    # 垂直调暗是它“看到”的像素 （kernel_size + 2）
    field_of_view = np.squeeze(np.uint(np.expand_dims(left_boundary, axis=1) + np.arange(expanded_kernel_width) - 1))

    # Assign weight to each pixel in the field of view. A matrix whos horizontal dim is the output pixels and the
    # vertical dim is a list of weights matching to the pixel in the field of view (that are specified in
    # 'field_of_view')
    # 为视野中的每个像素分配权重。水平变暗的矩阵是输出像素和
    # 垂直调暗是与视野中的像素匹配的权重列表（在'field_of_view'）
    weights = fixed_kernel(1.0 * np.expand_dims(match_coordinates, axis=1) - field_of_view - 1)

    # Normalize weights to sum up to 1. be careful from dividing by 0
    # 归一化权重以求和为 1。小心不要除以 0
    sum_weights = np.sum(weights, axis=1)
    sum_weights[sum_weights == 0] = 1.0
    weights = 1.0 * weights / np.expand_dims(sum_weights, axis=1)

    # We use this mirror structure as a trick for reflection padding at the boundaries
    # 我们使用这种镜像结构作为边界反射填充的技巧
    mirror = np.uint(np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))))
    field_of_view = mirror[np.mod(field_of_view, mirror.shape[0])]

    # Get rid of  weights and pixel positions that are of zero weight
    # 摆脱零权重的权重和像素位置
    non_zero_out_pixels = np.nonzero(np.any(weights, axis=0))
    weights = np.squeeze(weights[:, non_zero_out_pixels])
    field_of_view = np.squeeze(field_of_view[:, non_zero_out_pixels])

    # Final products are the relative positions and the matching weights, both are output_size X fixed_kernel_size
    # 最终产品是相对位置和匹配的重量，两者都是output_size X fixed_kernel_size
    return weights, field_of_view


def resize_along_dim(im, dim, weights, field_of_view):
    # To be able to act on each dim, we swap so that dim 0 is the wanted dim to resize
    # 为了能够对每个 dim 进行操作，我们交换，以便 dim 0 是想要调整大小的 dim
    tmp_im = np.swapaxes(im, dim, 0)

    # We add singleton dimensions to the weight matrix so we can multiply it with the big tensor we get for
    # tmp_im[field_of_view.T], (bsxfun style)
    # 我们将单例维度添加到权重矩阵中，以便我们可以将其与得到的大张量相乘
    # tmp_im[field_of_view.T]，（bsxfun风格）
    weights = np.reshape(weights.T, list(weights.T.shape) + (np.ndim(im) - 1) * [1])

    # This is a bit of a complicated multiplication: tmp_im[field_of_view.T] is a tensor of order image_dims+1.
    # for each pixel in the output-image it matches the positions the influence it from the input image (along 1 dim
    # only, this is why it only adds 1 dim to the shape). We then multiply, for each pixel, its set of positions with
    # the matching set of weights. we do this by this big tensor element-wise multiplication (MATLAB bsxfun style:
    # matching dims are multiplied element-wise while singletons mean that the matching dim is all multiplied by the
    # same number
    # 这是一个有点复杂的乘法：tmp_im[field_of_view。T] 是    image_dims + 1    阶张量。
    # 对于输出图像中的每个像素，它匹配的位置，影响它来自输入图像（沿 1 暗
    # 只是，这就是为什么它只在形状上增加1个暗淡）。然后，我们将每个像素的位置集乘以
    # 匹配的权重集。我们通过这个大的张量逐元素乘法（MATLAB bsxfun风格：
    # 匹配的 dim 是逐个乘法的，而单例意味着匹配的 dim 全部乘以
    # 相同的数字
    tmp_out_im = np.sum(tmp_im[field_of_view.T] * weights, axis=0)

    # Finally we swap back the axes to the original order
    # 最后，我们将轴交换回原始顺序
    return np.swapaxes(tmp_out_im, dim, 0)


def numeric_kernel(im, kernel, scale_factor, output_shape, kernel_shift_flag):
    # See kernel_shift function to understand what this is
    # 请参阅kernel_shift函数以了解这是什么
    if kernel_shift_flag:
        kernel = kernel_shift(kernel, scale_factor)

    # First run a correlation (convolution with flipped kernel)
    # 首先运行相关性（使用翻转内核进行卷积）
    out_im = np.zeros_like(im)
    for channel in range(np.ndim(im)):
        out_im[:, :, channel] = filters.correlate(im[:, :, channel], kernel)

    # Then subsample and return
    # 然后子采样并返回
    return out_im[np.round(np.linspace(0, im.shape[0] - 1 / scale_factor[0], output_shape[0])).astype(int)[:, None],
                  np.round(np.linspace(0, im.shape[1] - 1 / scale_factor[1], output_shape[1])).astype(int), :]


def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel:
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between od and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.
    # 移动内核有两个原因：
    # 1.质心不在内核的中心，这会产生歧义。没有办法知道
    # 退化过程包括移位，所以我们总是假设质心是内核的中心。
    # 2.我们进一步移动内核中心，使左上角的结果像素首先对应于 sfXsf 的中间
    # 像素。默认值是奇数大小位于第一个像素的中间，偶数大小的内核位于
    # 第一个像素的左上角。这就是为什么在OD和均匀尺寸之间需要不同的班次大小。
    # 鉴于这两个条件得到满足，我们很开心，很一致，测试方法如下：
    # 输入图像在插值（正则双立方）时与基本事实完全对齐。

    # First calculate the current center of mass for the kernel
    # 首先计算内核的当前质心
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second ("+ 0.5 * ....") is for applying condition 2 from the comments above
    # 第二个（“+ 0.5 * ....”）用于应用上述评论中的条件 2
    wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (sf - (kernel.shape[0] % 2))

    # Define the shift vector for the kernel shifting (x,y)
    # 定义内核移位的移位向量 （x，y）
    shift_vec = wanted_center_of_mass - current_center_of_mass

    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    # 在应用 shift 之前，我们首先填充内核，以便不会因移位而丢失任何内容
    #（最大偏移在暗光+1之间安全）
    kernel = np.pad(kernel, np.int(np.ceil(np.max(shift_vec))) + 1, 'constant')

    # Finally shift the kernel and return
    # 最后移动内核并返回
    return interpolation.shift(kernel, shift_vec)


# These next functions are all interpolation methods. x is the distance from the left pixel center
# 接下来的这些函数都是插值方法。x 是到左像素中心的距离


def cubic(x):
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((1.5*absx3 - 2.5*absx2 + 1) * (absx <= 1) +
            (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * ((1 < absx) & (absx <= 2)))


def lanczos2(x):
    return (((np.sin(pi*x) * np.sin(pi*x/2) + np.finfo(np.float32).eps) /
             ((pi**2 * x**2 / 2) + np.finfo(np.float32).eps))
            * (abs(x) < 2))


def box(x):
    return ((-0.5 <= x) & (x < 0.5)) * 1.0


def lanczos3(x):
    return (((np.sin(pi*x) * np.sin(pi*x/3) + np.finfo(np.float32).eps) /
            ((pi**2 * x**2 / 3) + np.finfo(np.float32).eps))
            * (abs(x) < 3))


def linear(x):
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))
