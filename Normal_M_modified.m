function[H] = Normal_M_modified(H)

[m_nN, m_nD] = size(H);
[rows, cols, vals] = find(H);

sum_row = sqrt(sum(H,2));
sum_col = sqrt(sum(H,1));

m_vMatsub = sum_row(rows).*sum_col(cols)';
H = sparse(rows, cols, vals./m_vMatsub, m_nN, m_nD);
