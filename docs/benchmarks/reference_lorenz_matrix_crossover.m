% Independent CPU reference only: exact matrices and crossed solves.
args=argv(); root=args{1}; output=args{2}; mode=args{3};
addpath(fullfile(root,'.localresources/code/octave-statistics-1.5.0/inst'));
addpath(fullfile(root,'.localresources/code/octave-statistics-1.5.0/inst/distributions'));
data=load(fullfile(output,'input.mat'));
if strcmp(mode,'form')
  count=size(data.x,1); n=size(data.x,2); d=size(data.x,3); width=45;
  native_design=zeros(count,n,width); native_gram=zeros(count,width,width);
  native_rhs=zeros(count,width); native_beta=zeros(count,width);
  design_singular_values=zeros(count,width); gram_rcond=zeros(count,1);
  direct_beta=zeros(count,width); centered_qr_beta=zeros(count,width);
  standardized_singular_values=zeros(count,width);
  for k=1:count
    X=squeeze(data.x(k,:,:)); Value=data.target(k,:)';
    D=x2fx(X,'quadratic'); G=D'*D; r=D'*Value; beta=G\r;
    assert(isequal(beta,data.captured_beta(k,:)'));
    native_design(k,:,:)=D; native_gram(k,:,:)=G; native_rhs(k,:)=r';
    native_beta(k,:)=beta'; design_singular_values(k,:)=svd(D)';
    gram_rcond(k)=rcond(G); direct_beta(k,:)=(D\Value)';
    mu=mean(X,1); scale=sqrt(mean(bsxfun(@minus,X,mu).^2,1));
    Z=bsxfun(@rdivide,bsxfun(@minus,X,mu),scale); DZ=x2fx(Z,'quadratic');
    [Q,R]=qr(DZ,0); ybar=mean(Value); bz=R\(Q'*(Value-ybar)); bz(1)=bz(1)+ybar;
    standardized_singular_values(k,:)=svd(DZ)';
    az=diag(bz(end-d+1:end)); idx=1+d;
    for i=1:d
      for j=(i+1):d
        idx=idx+1; az(i,j)=0.5*bz(idx); az(j,i)=az(i,j);
      end
    end
    a=az./(scale'*scale); linear=bz(2:d+1)./scale';
    b=linear-2*a*mu'; c=bz(1)-linear'*mu'+mu*a*mu';
    br=zeros(width,1); br(1)=c; br(2:d+1)=b; idx=1+d;
    for i=1:d
      for j=(i+1):d
        idx=idx+1; br(idx)=2*a(i,j);
      end
    end
    br(end-d+1:end)=diag(a); centered_qr_beta(k,:)=br';
  end
  octave_version=version(); x2fx_path=which('x2fx');
  save('-mat7-binary',fullfile(output,'reference.mat'),'native_design','native_gram',...
    'native_rhs','native_beta','design_singular_values','gram_rcond','direct_beta',...
    'centered_qr_beta','standardized_singular_values','octave_version','x2fx_path');
  fprintf('Exact captured regression replay: %d/%d\n',count,count);
elseif strcmp(mode,'solve')
  count=size(data.native_gram,1); width=size(data.native_gram,2);
  octave_cross_beta=zeros(4,count,width); lu_cross_beta=zeros(4,count,width);
  chol_cross_beta=nan(4,count,width); chol_status=zeros(4,count);
  for arm=1:4
    for k=1:count
      if arm<=2; G=squeeze(data.native_gram(k,:,:)); else; G=squeeze(data.tf_gram(k,:,:)); end
      if mod(arm,2)==1; r=data.native_rhs(k,:)'; else; r=data.tf_rhs(k,:)'; end
      octave_cross_beta(arm,k,:)=(G\r)';
      [L,U,P]=lu(G); lu_cross_beta(arm,k,:)=(U\(L\(P*r)))';
      [R,p]=chol(G); chol_status(arm,k)=p;
      if p==0; chol_cross_beta(arm,k,:)=(R\(R'\r))'; end
    end
  end
  assert(isequal(squeeze(octave_cross_beta(1,:,:)),data.native_beta));
  save('-mat7-binary',fullfile(output,'crosssolve.mat'),'octave_cross_beta',...
    'lu_cross_beta','chol_cross_beta','chol_status');
  fprintf('Crossed original/TF matrices: %d solves\n',4*count);
elseif strcmp(mode,'fallback')
  count=size(data.native_gram,1); width=size(data.native_gram,2);
  ordinary_beta=zeros(4,count,width); forced_singular_beta=zeros(4,count,width);
  gram_spectrum=zeros(4,count,width); gram_rcond=zeros(4,count);
  warning_ids=cell(4,count); cached_types=cell(4,count);
  for arm=1:4
    for k=1:count
      if arm<=2; G=squeeze(data.native_gram(k,:,:)); else; G=squeeze(data.tf_gram(k,:,:)); end
      if mod(arm,2)==1; r=data.native_rhs(k,:)'; else; r=data.tf_rhs(k,:)'; end
      G=matrix_type(G,'unknown'); lastwarn(''); beta=G\r;
      [unused_message,warning_id]=lastwarn(); warning_ids{arm,k}=warning_id;
      cached_types{arm,k}=matrix_type(G,'nocompute'); ordinary_beta(arm,k,:)=beta';
      forced_singular_beta(arm,k,:)=(matrix_type(G,'singular')\r)';
      gram_spectrum(arm,k,:)=svd(G)'; gram_rcond(arm,k)=rcond(G);
    end
  end
  assert(isequal(squeeze(ordinary_beta(1,:,:)),data.native_beta));
  octave_version=version();
  save('-mat7-binary',fullfile(output,'fallback.mat'),'ordinary_beta',...
    'forced_singular_beta','gram_spectrum','gram_rcond','warning_ids','cached_types',...
    'octave_version');
  fprintf('Ordinary and forced minimum-norm solves: %d pairs\n',4*count);
else
  error('Unknown diagnostic mode');
end
