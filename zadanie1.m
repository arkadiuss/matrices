A = ones(10,10);
function[time, res] = time_measure(fun)
  start_time = clock();
  res = fun();
  time = etime(clock(), start_time)*1e3;
endfunction

function[result] = mul_ijk(A,B)
  result = zeros(rows(A), columns(B));
  for i = 1:rows(A)
    for j = 1:columns(B)
      for k = 1:columns(A)
        result(i,j) += A(i,k) + B(k,j);
      endfor
    endfor
  endfor
endfunction

function[result] = mul_ikj(A,B)
  result = zeros(rows(A), columns(B));
  for i = 1:rows(A)
    for k = 1:columns(A)
      for j = 1:columns(B)
        result(i,j) += A(i,k) + B(k,j);
      endfor
    endfor
  endfor
endfunction

function[result] = mul_jik(A,B)
  result = zeros(rows(A), columns(B));
  for j = 1:columns(B)
    for i = 1:rows(A)
      for k = 1:columns(A)
        result(i,j) += A(i,k) + B(k,j);
      endfor
    endfor
  endfor
endfunction

function[result] = mul_jki(A,B)
  result = zeros(rows(A), columns(B));
  for j = 1:columns(B)
    for k = 1:columns(A)
      for i = 1:rows(A)
        result(i,j) += A(i,k) + B(k,j);
      endfor
    endfor
  endfor
endfunction

function[result] = mul_kij(A,B)
  result = zeros(rows(A), columns(B));
  for k = 1:columns(A)
    for i = 1:rows(A)
      for j = 1:columns(B)
        result(i,j) += A(i,k) + B(k,j);
      endfor
    endfor
  endfor
endfunction

function[result] = mul_kji(A,B)
  result = zeros(rows(A), columns(B));
  for k = 1:columns(A)
    for j = 1:columns(B)
      for i = 1:rows(A)
        result(i,j) += A(i,k) + B(k,j);
      endfor
    endfor
  endfor
endfunction

X = [ 10, 30, 100 ];
Yijk = [];
Yikj = [];
Yjik = [];
Yjki = [];
Ykij = [];
Ykji = [];
for x = X
  A = ones(x,x);
  B = ones(x,x);
  [Yijk(end + 1), res] = time_measure(@() mul_ijk(A,B));
  [Yikj(end + 1), res] = time_measure(@() mul_ikj(A,B));
  [Yjik(end + 1), res] = time_measure(@() mul_jik(A,B));
  [Yjki(end + 1), res] = time_measure(@() mul_jki(A,B));
  [Ykij(end + 1), res] = time_measure(@() mul_kij(A,B));
  [Ykji(end + 1), res] = time_measure(@() mul_kji(A,B));
endfor

plot(X, Yijk, ";ijk;", X, Yikj, ";ikj;", X, Yjik, ";jik;",  X, Yjki, ";jki;",  X, Ykij, ";kij;", X, Ykji, ";kji;");