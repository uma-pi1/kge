class BaseModel:
  """
  Implements basic embedding layer
  """

  def encode_subject(self, s):
    pass

  def encode_relation(self, r):
    pass

  def encode_object(self, o):
    pass

  def forward_one_to_one(self, s, r, o):
    raise NotImplemented

  def forward_one_to_n_sr(self, s, r):
    raise NotImplemented

  def forward_one_to_n_ro(self, r, o):
    raise NotImplemented

  def forward_n_to_n(self, r):
    raise NotImplemented


class DistMult(BaseModel):
  """
  Implements the 1:1, 1:N, N:N forward functions
  """


