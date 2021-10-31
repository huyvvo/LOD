function [ap,pr,rc,pos,dc] = x2pr(proposals, bboxes, x, thresh)
%
% [ap, pr,rc,pos,dc] = x2pr(proposals, bboxes, x, thresh) 
%

  if isempty(gcp('nocreate'))
    [ap,pr,rc,pos,dc] = x2pr_seq(proposals, bboxes, x, thresh);
  else 
    [ap,pr,rc,pos,dc] = x2pr_par(proposals, bboxes, x, thresh);
  end
end

function [ap,pr,rc,pos,dc] = x2pr_par(proposals, bboxes, x, thresh)
  n = numel(x);
  pos = cell(n,1);
  dc = cell(n,1);
  parfor i = 1:n 
    if isempty(proposals{i}) || isempty(x{i})
      pos{i} = [0];
      dc{i} = [0];
    else
      IoU = pairwise_bbox_iou(proposals{i}(x{i},:), bboxes{i});
      dc{i} = compute_dc(IoU,thresh);
      pos{i} = compute_pos(IoU,thresh);
    end
  end

  num_bboxes = cellfun(@(el) size(el,1), bboxes);

  K = max(cellfun(@numel,x));
  pr = cell(1,K);
  rc = cell(1,K);
  parfor i = 1:K
    num_pos = sum(cellfun(@(el) el(min(i,end)), pos));
    num_props = sum(cellfun(@(el) min(i,numel(el)), pos));
    pr{i} = num_pos/num_props;
    sum_dc = sum(cellfun(@(el) el(min(i,end)), dc));
    rc{i} = sum_dc/sum(num_bboxes);
  end  
  pr = cell2mat(pr);
  rc = cell2mat(rc);
  ap = sum(pr.*(rc-[0 rc(1:end-1)]));
end

function [ap,pr,rc,pos,dc] = x2pr_seq(proposals, bboxes, x, thresh)
  n = numel(x);
  pos = cell(n,1);
  dc = cell(n,1);
  for i = 1:n 
    if isempty(proposals{i}) || isempty(x{i})
      pos{i} = [0];
      dc{i} = [0];
    else
      IoU = pairwise_bbox_iou(proposals{i}(x{i},:), bboxes{i});
      dc{i} = compute_dc(IoU,thresh);
      pos{i} = compute_pos(IoU,thresh);
    end
  end

  num_bboxes = cellfun(@(el) size(el,1), bboxes);

  K = max(cellfun(@numel,x));
  pr = cell(1,K);
  rc = cell(1,K);
  for i = 1:K
    num_pos = sum(cellfun(@(el) el(min(i,end)), pos));
    num_props = sum(cellfun(@(el) min(i,numel(el)), pos));
    pr{i} = num_pos/num_props;
    sum_dc = sum(cellfun(@(el) el(min(i,end)), dc));
    rc{i} = sum_dc/sum(num_bboxes);
  end  
  pr = cell2mat(pr);
  rc = cell2mat(rc);
  ap = sum(pr.*(rc-[0 rc(1:end-1)]));
end

function [dc] = compute_dc(IoU, thresh)
  k = size(IoU,1);
  dc = zeros(1,k);
  max_IoU = zeros(1,size(IoU,2));
  for i = 1:k
    max_IoU = max(max_IoU, IoU(i,:));
    dc(i) = sum(max_IoU >= thresh);
  end
end

function [pos] = compute_pos(IoU, thresh)
  if size(IoU,1) > 1
    for i = 1:size(IoU,2)
      ids = find(IoU(:,i) >= thresh);
      IoU(ids(2:end),i) = 0;
    end
  end
  pos = cumsum(max(IoU,[],2)>=thresh);
  pos = pos(:)';
end