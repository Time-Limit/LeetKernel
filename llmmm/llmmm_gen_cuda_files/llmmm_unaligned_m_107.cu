#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_107_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<107> mm_instantiator;

public:
  UnalignedM_107_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_107_MMInstantiatorWrapper__;

}  // namespace LLMMM
