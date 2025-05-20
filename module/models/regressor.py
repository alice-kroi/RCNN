import torch.nn as nn
import torch

class RCNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=4096, output_dim=4):
        """
        边界框回归器模块
        参数：
            input_dim: 输入特征维度 (VGG16特征提取后的维度512*7*7)
            hidden_dim: 全连接层隐藏维度 (默认4096)
            output_dim: 输出维度 (固定为4个坐标偏移量)
        """
        super(RCNNRegressor, self).__init__()
        
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim)  # 输出坐标偏移量 (dx, dy, dw, dh)
        )
        
        self._init_weights()

    def _init_weights(self):
        """ Xavier初始化 """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        """
        输入：
            x: 特征张量 [batch_size, input_dim]
        输出：
            deltas: 边界框偏移量 [batch_size, 4]
        """
        return self.regressor(x)
    
    def get_bbox(self, rois, deltas):
        """
        根据回归器输出的偏移量计算边界框
        参数：
            rois: 区域提议列表 [N, 4] (x1, y1, x2, y2)
            deltas: 回归器输出的偏移量 [N, 4] (dx, dy, dw, dh)
        返回：
            bboxes: 修正后的边界框 [N, 4] (x1, y1, x2, y2)
        """
        # 计算中心点和宽高
        width = rois[:, 2] - rois[:, 0] + 1.0  # 宽度
        height = rois[:, 3] - rois[:, 1] + 1.0  # 高度
        ctr_x = rois[:, 0] + 0.5 * width  # 中心点x
        ctr_y = rois[:, 1] + 0.5 * height  # 中心点y

        # 应用偏移量
        dx = deltas[:, 0] * width  # 偏移x
        dy = deltas[:, 1] * height  # 偏移y

        # 修正中心点
        pred_ctr_x = ctr_x + dx  # 预测中心点x
        pred_ctr_y = ctr_y + dy  # 预测中心点y

        # 修正宽高
        pred_w = torch.exp(deltas[:, 2]) * width  # 预测宽度
        pred_h = torch.exp(deltas[:, 3]) * height  # 预测高度
        # 计算边界框
        pred_x1 = pred_ctr_x - 0.5 * pred_w  # 边界框x1
        pred_y1 = pred_ctr_y - 0.5 * pred_h  # 边界框y1
        pred_x2 = pred_ctr_x + 0.5 * pred_w  # 边界框x2
        pred_y2 = pred_ctr_y + 0.5 * pred_h  # 边界框y2

        # 堆叠边界框
        bboxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)
        return bboxes

    def nms(self, bboxes, scores, threshold=0.5):
        """
        非极大值抑制
        参数：
            bboxes: 边界框列表 [N, 4] (x1, y1, x2, y2)
            scores: 对应的置信度 [N]
            threshold: IoU阈值
        返回：
            keep: 保留的边界框索引
        """
        # 转换为Tensor
        bboxes = bboxes.clone().detach() if isinstance(bboxes, torch.Tensor) else torch.tensor(bboxes)
        scores = scores.clone().detach() if isinstance(scores, torch.Tensor) else torch.tensor(scores)

        if scores.dim() > 1:
            scores = scores.squeeze(1)  # 处理多类别分数
        # 计算边界框面积
        areas = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1)

        # 按置信度降序排序
        _, order = scores.sort(0, descending=True)
        #print(order)
        keep = []  # 保留的边界框索引
        while order.numel() > 0:
            i = order[0].item() if order.dim() > 0 else order.item()
            keep.append(i)

            if order.numel() == 1:  # 只剩下一个边界框
                break

            # 计算IoU
            xx1 = bboxes[order[1:], 0].clamp(min=bboxes[i, 0])  # 计算x1
            yy1 = bboxes[order[1:], 1].clamp(min=bboxes[i, 1])  # 计算y1
            xx2 = bboxes[order[1:], 2].clamp(max=bboxes[i, 2])  # 计算x2
            yy2 = bboxes[order[1:], 3].clamp(max=bboxes[i, 3])  # 计算y2
            w = (xx2 - xx1).clamp(min=0)  # 计算宽度
            h = (yy2 - yy1).clamp(min=0)  # 计算高度

            inter = w * h  # 计算交集面积
            iou = inter / (areas[i] + areas[order[1:]] - inter)  # 计算IoU

            # 保留IoU小于阈值的边界框
            idx = (iou <= threshold).nonzero().squeeze()
            if idx.numel() == 0:  # 没有边界框
                break
            order = order[idx + 1]  # 更新边界框索引

        return keep
    def post_process(self, rois, scores, deltas, img_size, threshold=0.5, nms_threshold=0.5):
        """
        后处理函数
        参数：
            rois: 区域提议列表 [N, 4] (x1, y1, x2, y2)
            scores: 对应的置信度 [N, num_classes]
            deltas: 回归器输出的偏移量 [N, 4] (dx, dy, dw, dh)
            img_size: 原始图像尺寸 (H, W)
            threshold: 置信度阈值
            nms_threshold: NMS阈值
        返回：
            bboxes: 最终的边界框列表 [M, 5] (x1, y1, x2, y2, score)
        """
        # 应用回归器
        bboxes = self.get_bbox(rois, deltas)

        # 边界框裁剪
        bboxes[:, 0] = bboxes[:, 0].clamp(min=0, max=img_size[1])  # x1
        bboxes[:, 1] = bboxes[:, 1].clamp(min=0, max=img_size[0])  # y1
        bboxes[:, 2] = bboxes[:, 2].clamp(min=0, max=img_size[1])  # x2
        bboxes[:, 3] = bboxes[:, 3].clamp(min=0, max=img_size[0])  # y2

        # 置信度过滤
        keep = torch.where(scores > threshold)[0]  # 过滤
        bboxes = bboxes[keep]  # 边界框
        scores = scores[keep]  # 置信度

        # NMS
        keep = self.nms(bboxes, scores, nms_threshold)  # NMS
        bboxes = bboxes[keep]  # 边界框
        scores = scores[keep]  # 置信度

        # 堆叠结果
        bboxes = torch.cat([bboxes, scores.unsqueeze(1)], dim=1)  # 边界框和置信度
        return bboxes
    def forward_rois(self, x, rois):
        """
        输入：
            x: 特征张量 [batch_size, input_dim]
            rois: 区域提议列表 [N, 5] (batch_index, x1, y1, x2, y2)
        输出：
            deltas: 边界框偏移量 [N, 4]
        """
        deltas = self.regressor(x)  # 边界框偏移量
        return deltas[rois[:, 0].long()]  # 转换为long类型索引
    


__all__ = ['RCNNRegressor']

if __name__ == "__main__":
    # 完整测试示例
    model = RCNNRegressor(input_dim=25088)
    
    # 模拟输入数据
    batch_size = 2
    num_rois_per_image = 5  # 每张图像的ROI数量
    img_size = (600, 800)  # (H, W)
    
    # 生成测试数据（修正维度）
    features = torch.randn(batch_size, 25088)  # 特征张量
    rois = torch.stack([
        torch.randint(0, 400, (batch_size * num_rois_per_image,)),  # x1
        torch.randint(0, 300, (batch_size * num_rois_per_image,)),  # y1
        torch.randint(400, 800, (batch_size * num_rois_per_image,)),# x2
        torch.randint(300, 600, (batch_size * num_rois_per_image,)) # y2
    ], dim=1).float()
    
    # 添加batch索引到ROIs
    batch_indices = torch.repeat_interleave(torch.arange(batch_size), num_rois_per_image)
    rois = torch.cat([batch_indices.unsqueeze(1).float(), rois], dim=1)  # [batch_size*num_rois_per_image, 5]
    
    # 使用forward_rois获取正确的deltas
    deltas = model.forward_rois(features, rois)
    print("回归器输出形状:", deltas.shape)  # 应输出 [10, 4]
    
    # 边界框生成测试 (修正参数)
    pred_boxes = model.get_bbox(rois[:, 1:], deltas)  # 使用ROIs的坐标部分
    print("\n预测框示例:")
    print(pred_boxes[:3])
    
    # 后处理测试 (修正参数)
    class_scores = torch.sigmoid(torch.randn(len(rois), 20))  # 分数数量与ROIs数量一致
    max_scores, max_indices = torch.max(class_scores, dim=1)  # 获取每个ROI的最大得分和类别索引
    print("\n最大得分示例:")
    print(max_scores[:3])
    print("\n类别索引示例:")
    print(max_indices[:3])
    final_boxes = model.post_process(
    rois=rois[:, 1:],
    scores=max_scores,  # 使用最大得分
    deltas=deltas,
    img_size=img_size,
    threshold=0.7,
    nms_threshold=0.3
    )
    
    print("\n后处理结果:")
    print(f"保留边界框数量: {len(final_boxes)}")